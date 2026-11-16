"""End-to-end orchestration: train -> (new data) -> retrain -> score -> triage -> work orders."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agent.work_orders import WorkOrderStore, render_markdown
from .data.loader import load_incoming, load_subset, merge_batches, read_cmapss_file, split_engines, RAW_COLS
from .mlops.drift import drift_report
from .mlops.registry import ModelRegistry
from .mlops.tracking import Tracker
from .scoring import score_fleet, score_history, triage_rows
from .training import save_tables, train_bundle

log = logging.getLogger(__name__)


class FleetGuard:
    def __init__(self, cfg, track: bool = True, fast: bool = False, candidates: list[str] | None = None):
        self.cfg = cfg
        self.tracker = Tracker(cfg.tracking_uri, enabled=track)
        self.fast = fast
        self.candidates = candidates

    # ------------------------------------------------------------ helpers
    def work_orders(self) -> WorkOrderStore:
        """The work-order store (SQLite). Imports an older work_orders.json once, if present."""
        return WorkOrderStore(self.cfg.path("database"), import_json=self.cfg.path("outputs") / "work_orders.json")

    def registry(self, subset: str) -> ModelRegistry:
        return ModelRegistry(self.cfg.path("models"), subset)

    def out_dir(self, subset: str) -> Path:
        d = self.cfg.path("outputs") / subset
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self, subset: str) -> Path:
        return self.cfg.path("models") / subset / "data_manifest.json"

    def _manifest(self, subset: str) -> dict:
        p = self._manifest_path(subset)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"base_engines": None, "held_back_engines": [], "ingested": []}

    def _save_manifest(self, subset: str, manifest: dict) -> None:
        p = self._manifest_path(subset)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _training_data(self, subset: str, data: dict, manifest: dict) -> tuple[pd.DataFrame, list[str]]:
        """Base training engines + every batch ingested so far."""
        train = data["train"]
        if manifest["base_engines"] is not None:
            train = train[train["engine_id"].isin(manifest["base_engines"])]
        sources = [f"train_{subset}.txt ({train['engine_id'].nunique()} engines)"]
        archive = self.cfg.path("incoming_data") / "archive"
        for name in manifest["ingested"]:
            batch = read_cmapss_file(archive / name) if not name.endswith(".csv") else pd.read_csv(archive / name)[RAW_COLS]
            train = merge_batches(train, batch)
            sources.append(f"{name} ({batch['engine_id'].nunique()} engines)")
        return train.reset_index(drop=True), sources

    # -------------------------------------------------------------- train
    def train(self, subset: str, initial_fraction: float = 1.0) -> dict:
        """Train a model version and promote it if it is the first one or beats the champion."""
        subset = subset.upper()
        data = load_subset(self.cfg.path("raw_data"), subset)
        manifest = self._manifest(subset)
        if initial_fraction < 1.0:
            keep, held = split_engines(data["train"]["engine_id"].unique(), 1 - initial_fraction, self.cfg["seed"] + 1)
            manifest = {"base_engines": keep.tolist(), "held_back_engines": held.tolist(), "ingested": []}
            self._save_manifest(subset, manifest)
        train, sources = self._training_data(subset, data, manifest)
        return self._train_and_decide(subset, data, train, sources, reason="initial training")

    def _train_and_decide(self, subset, data, train, sources, reason: str, drift: pd.DataFrame | None = None) -> dict:
        reg = self.registry(subset)
        version = reg.next_version()
        log.info("[%s] training %s on %d engines", subset, version, train["engine_id"].nunique())
        bundle, meta, tables = train_bundle(
            self.cfg, subset, train, data["test"], data["test_rul"], version, self.tracker,
            candidates=self.candidates, fast=self.fast, data_sources=sources,
        )
        meta["trigger"] = reason
        if drift is not None:
            meta["drift_on_new_data"] = drift.head(10).to_dict(orient="records")
        reg.register(bundle, meta)

        # Champion / challenger: compare on the same fixed hold-out (official test set).
        champion = reg.champion
        tol = self.cfg["retraining"]["promotion_tolerance"]
        new_rmse = bundle["metrics"]["test_rmse"]
        if champion is None:
            decision = f"promoted: first model ({reason})"
            reg.promote(version, decision)
        else:
            old = next(v for v in reg.versions() if v["version"] == champion)
            old_rmse = old["metrics"]["test_rmse"]
            if new_rmse <= old_rmse + tol:
                decision = f"promoted: test RMSE {new_rmse:.2f} vs champion {champion} {old_rmse:.2f}"
                reg.promote(version, decision)
            else:
                decision = f"rejected: test RMSE {new_rmse:.2f} worse than champion {champion} {old_rmse:.2f}"
                reg.reject(version, decision)
        if reg.champion == version:
            self.tracker.set_champion(subset, bundle.get("mlflow_model_version"))
            save_tables(tables, self.out_dir(subset) / "evaluation", meta)
        log.info("[%s] %s %s", subset, version, decision)
        return {"subset": subset, "version": version, "decision": decision, "metrics": bundle["metrics"]}

    # ---------------------------------------------------- new data / retrain
    def simulate_arrival(self, subset: str) -> Path | None:
        """Demo helper: the engines held back from v1 'arrive' as a new run-to-failure batch."""
        subset = subset.upper()
        manifest = self._manifest(subset)
        if not manifest["held_back_engines"]:
            log.info("[%s] no held-back engines to release", subset)
            return None
        data = load_subset(self.cfg.path("raw_data"), subset)
        batch = data["train"][data["train"]["engine_id"].isin(manifest["held_back_engines"])]
        incoming = self.cfg.path("incoming_data")
        incoming.mkdir(parents=True, exist_ok=True)
        n = len(list(incoming.glob(f"{subset}_*"))) + len(manifest["ingested"]) + 1
        path = incoming / f"{subset}_batch{n:02d}.csv"
        batch.to_csv(path, index=False)
        manifest["held_back_engines"] = []
        self._save_manifest(subset, manifest)
        log.info("[%s] new batch with %d engines written to %s", subset, batch["engine_id"].nunique(), path)
        return path

    def retrain(self, subset: str, force: bool = False) -> dict:
        """Retraining flow triggered by new data in data/incoming/."""
        subset = subset.upper()
        incoming_dir = self.cfg.path("incoming_data")
        batches = load_incoming(incoming_dir, subset)
        if not batches and not force:
            return {"subset": subset, "decision": "skipped: no new data in data/incoming"}
        data = load_subset(self.cfg.path("raw_data"), subset)
        manifest = self._manifest(subset)

        drift = None
        reg = self.registry(subset)
        if batches and reg.champion:
            champion = reg.load()
            new_feats = champion["features"].transform(pd.concat(batches, ignore_index=True))
            cols = champion["reference"].columns.tolist()
            drift = drift_report(champion["reference"], new_feats, cols, self.cfg["retraining"]["psi_alert"])
            drift.to_csv(self.out_dir(subset) / "drift_new_data.csv", index=False)

        archive = incoming_dir / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        for b in batches:
            shutil.move(str(incoming_dir / b.attrs["source"]), str(archive / b.attrs["source"]))
            manifest["ingested"].append(b.attrs["source"])
        self._save_manifest(subset, manifest)

        train, sources = self._training_data(subset, data, manifest)
        n_drift = int((drift["status"] == "drift").sum()) if drift is not None else 0
        reason = f"new data: {', '.join(b.attrs['source'] for b in batches) or 'forced'}; {n_drift} drifted features"
        result = self._train_and_decide(subset, data, train, sources, reason=reason, drift=drift)
        result["drifted_features"] = n_drift
        return result

    # --------------------------------------------------------------- score
    def score(self, subset: str) -> pd.DataFrame:
        """Score the current fleet (official test engines) and raise work orders."""
        subset = subset.upper()
        bundle = self.registry(subset).load()
        data = load_subset(self.cfg.path("raw_data"), subset)
        from .data.loader import add_test_rul

        fleet_raw = add_test_rul(data["test"], data["test_rul"])  # true RUL kept only for evaluation columns
        store = self.work_orders()
        fleet, hist, quality = score_fleet(bundle, fleet_raw, store)
        store.save()
        run_id = store.db.record_scoring(subset, bundle["version"], fleet)

        out = self.out_dir(subset)
        fleet.to_csv(out / "fleet_latest.csv", index=False)
        hist.to_csv(out / "fleet_history.csv.gz", index=False, compression="gzip")
        wo_dir = out / "work_orders"
        if wo_dir.exists():
            shutil.rmtree(wo_dir)
        wo_dir.mkdir()
        for wo in store.orders:
            if wo["subset"] == subset:
                (wo_dir / f"{wo['wo_id']}.md").write_text(render_markdown(wo), encoding="utf-8")

        # Monitoring: input drift of the live fleet vs training reference, prediction summary.
        feats = bundle["features"].transform(fleet_raw)
        ref = bundle["reference"]
        drift = drift_report(ref, feats, ref.columns.tolist(), self.cfg["retraining"]["psi_alert"])
        drift.to_csv(out / "drift_fleet.csv", index=False)
        monitoring = {
            "subset": subset,
            "model_version": bundle["version"],
            "scoring_run_id": run_id,
            "scored_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "engines": int(len(fleet)),
            "severity_counts": fleet["severity"].value_counts().to_dict(),
            "mean_rul_pred": float(fleet["rul_pred"].mean()),
            "alarm_rate": float(fleet["anomaly_alarm"].mean()),
            "data_quality": quality,
            "drifted_features": drift[drift["status"] == "drift"]["feature"].tolist(),
            "max_psi": float(drift["psi"].max()),
        }
        (out / "monitoring.json").write_text(json.dumps(monitoring, indent=2), encoding="utf-8")
        log.info("[%s] scored %d engines with %s: %s", subset, len(fleet), bundle["version"], monitoring["severity_counts"])
        return fleet

    # -------------------------------------------------------------- stream
    def stream(self, subset: str, step: int = 10) -> pd.DataFrame:
        """Replay the fleet cycle by cycle, as if telemetry arrived live.

        At every tick each engine has revealed its readings up to that cycle.
        The agent runs on each engine's current state and the work-order store
        de-duplicates and escalates exactly as it would in production.
        """
        subset = subset.upper()
        bundle = self.registry(subset).load()
        data = load_subset(self.cfg.path("raw_data"), subset)
        hist, feats, _ = score_history(bundle, data["test"])
        out = self.out_dir(subset)
        store_path = out / "stream_work_orders.json"
        if store_path.exists():
            store_path.unlink()
        store = WorkOrderStore(store_path)
        events = []
        last_cycle = hist.groupby("engine_id")["cycle"].transform("max")
        max_tick = int(hist["cycle"].max())
        for tick in list(range(step, max_tick, step)) + [max_tick]:
            # Engines whose telemetry ends before this tick have stopped reporting; skip them.
            at_tick = hist[(hist["cycle"] == np.minimum(tick, last_cycle)) & (last_cycle >= tick - step + 1)]
            for d in triage_rows(bundle, hist, feats, at_tick.index):
                event, wo = store.apply(subset, d, bundle["version"])
                if event in ("CREATED", "ESCALATED"):
                    events.append({"tick": tick, "engine_id": d.engine_id, "event": event, "severity": d.severity,
                                   "rul_pred": d.rul_pred, "wo_id": wo["wo_id"], "reasons": "|".join(d.reason_codes)})
        store.save()
        ev = pd.DataFrame(events, columns=["tick", "engine_id", "event", "severity", "rul_pred", "wo_id", "reasons"])
        ev.to_csv(out / "stream_events.csv", index=False)
        log.info("[%s] stream replay: %d ticks, %d created/escalated events, %d work orders",
                 subset, len(range(step, max_tick, step)) + 1, len(ev), len(store.orders))
        return ev

    # ------------------------------------------------------------- run all
    def run_all(self, subsets, fresh: bool = True, stream_step: int = 10) -> dict:
        """The full demo for each subset:
        v1 on 70% of engines -> new batch arrives -> drift check + retrain v2 ->
        champion/challenger decision -> score fleet -> work orders -> streaming replay."""
        summary = {}
        if fresh:
            self.reset(subsets)
        for subset in subsets:
            t0 = time.time()
            frac = self.cfg["retraining"]["initial_engine_fraction"]
            first = self.train(subset, initial_fraction=frac)
            self.simulate_arrival(subset)
            second = self.retrain(subset)
            fleet = self.score(subset)
            ev = self.stream(subset, step=stream_step)
            champion = self.registry(subset).load()
            summary[subset] = {
                "v1": {"decision": first["decision"], "test_rmse": first["metrics"]["test_rmse"]},
                "v2": {"decision": second["decision"], "test_rmse": second["metrics"]["test_rmse"],
                       "drifted_features": second.get("drifted_features")},
                "champion": champion["version"],
                "metrics": {k: champion["metrics"][k] for k in (
                    "test_rmse", "test_mae", "test_nasa_score", "test_uncapped_rmse", "test_interval_coverage",
                    "agent_recall_near_failure", "agent_false_alarm_rate", "agent_missed_critical")},
                "severity_counts": fleet["severity"].value_counts().to_dict(),
                "stream_events": int(len(ev)),
                "seconds": round(time.time() - t0, 1),
            }
        path = self.cfg.path("outputs") / "summary.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        existing.update(summary)
        path.write_text(json.dumps(existing, indent=2, default=float), encoding="utf-8")
        return summary

    def reset(self, subsets) -> None:
        """Remove previous models/outputs/work orders for these subsets (fresh demo)."""
        for subset in subsets:
            for d in (self.cfg.path("models") / subset, self.cfg.path("outputs") / subset):
                if d.exists():
                    shutil.rmtree(d)
            inc = self.cfg.path("incoming_data")
            for p in list(inc.glob(f"{subset}_*")) + list((inc / "archive").glob(f"{subset}_*")):
                p.unlink()
        self.work_orders().remove_fleets(subsets)
