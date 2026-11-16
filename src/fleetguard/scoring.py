"""Scoring a fleet with a trained bundle and handing the results to the agent."""

from __future__ import annotations

import pandas as pd

from .agent.triage import TriageAgent
from .agent.work_orders import WorkOrderStore


def score_history(bundle: dict, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Featurise + predict every row of every engine.

    Features are causal, so the row at cycle t is exactly what the system
    would have predicted at time t. Returns (row-level history, features, quality report).
    """
    fp = bundle["features"]
    feats, report = fp.transform(raw, return_report=True)
    preds = bundle["model"].predict(feats)
    scores = bundle["anomaly"].score(feats)
    keep = ["engine_id", "cycle", "condition"] + [c for c in ("rul",) if c in feats.columns]
    hist = feats[keep].join(preds).join(scores)
    g = feats.groupby("engine_id")
    hist["n_cycles"] = g.cumcount() + 1
    window = max(fp.windows)
    hist["imputed_fraction"] = (
        (feats["imputed"] > 0).astype(float).groupby(feats["engine_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    )
    return hist, feats, report.to_dict()


def triage_rows(bundle: dict, hist: pd.DataFrame, feats: pd.DataFrame, rows) -> list:
    """Run the agent on selected row indices (e.g. the latest row per engine)."""
    agent = TriageAgent(bundle["thresholds"])
    detector = bundle["anomaly"]
    decisions = []
    for i in rows:
        state = hist.loc[i].to_dict()
        decisions.append(agent.decide(state, detector.sensor_deviation(feats.loc[i])))
    return decisions


def score_fleet(bundle: dict, raw: pd.DataFrame, store: WorkOrderStore | None = None, fleet_label: str | None = None):
    """Batch mode: latest state per engine -> decision -> work order.

    ``fleet_label`` names the fleet in the work-order store (defaults to the
    model's subset); uploaded fleets use their own label so engine IDs never
    collide with the benchmark fleet.
    """
    hist, feats, report = score_history(bundle, raw)
    latest_idx = hist.groupby("engine_id")["cycle"].idxmax().sort_values().tolist()
    decisions = triage_rows(bundle, hist, feats, latest_idx)
    events = []
    if store is not None:
        for d in decisions:
            event, wo = store.apply(fleet_label or bundle["subset"], d, bundle["version"])
            events.append((event, wo["wo_id"] if wo else None))
    fleet = _fleet_table(hist.loc[latest_idx].reset_index(drop=True), decisions, events)
    return fleet, hist, report


def _fleet_table(latest: pd.DataFrame, decisions, events) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(decisions):
        row = {
            "engine_id": d.engine_id, "cycle": d.cycle, "severity": d.severity,
            "rul_pred": d.rul_pred, "rul_lower": d.rul_lower, "rul_upper": d.rul_upper,
            "anomaly_index": d.anomaly_index, "anomaly_alarm": d.anomaly_alarm,
            "due_within_cycles": d.due_within_cycles, "action": d.action,
            "reason_codes": "|".join(d.reason_codes),
            "indicators": ", ".join(f"{x['name']} {x['deviation_sd']:+.1f}SD" for x in d.indicators),
            "imputed_fraction": round(float(latest.loc[i, "imputed_fraction"]), 3),
        }
        if "rul" in latest.columns:
            row["true_rul"] = int(latest.loc[i, "rul"])
        if events:
            row["work_order_event"], row["wo_id"] = events[i]
        rows.append(row)
    order = {"CRITICAL": 0, "HIGH": 1, "REVIEW": 2, "MEDIUM": 3, "LOW": 4}
    df = pd.DataFrame(rows)
    df["_o"] = df["severity"].map(order)
    return df.sort_values(["_o", "rul_pred"]).drop(columns="_o").reset_index(drop=True)
