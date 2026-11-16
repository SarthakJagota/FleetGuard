"""Training: one call builds a complete, versioned FleetGuard bundle for a subset.

    raw run-to-failure data
      -> engine-level train / validation split (no leakage between engines)
      -> FeaturePipeline.fit on training engines only
      -> candidate RUL models (with imbalance weights)    -> MLflow nested runs
      -> best model by validation RMSE
      -> conformal intervals on validation snapshots
      -> anomaly detectors fitted on healthy early-life data
      -> triage threshold calibration on validation snapshots
      -> evaluation on the official NASA test set
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .agent.triage import TriageAgent, TriageThresholds, calibrate_high_threshold
from .data.features import FeaturePipeline, last_rows, truncation_snapshots
from .data.loader import add_test_rul, add_train_rul, split_engines
from .mlops.drift import reference_sample
from .mlops.tracking import Tracker
from .models.anomaly import AnomalyDetector
from .models.metrics import agent_metrics, error_by_bucket, interval_metrics, regression_metrics, warning_metrics
from .models.risk import FailureRisk, calibration_table, risk_metrics
from .models.rul import (
    MODEL_DESCRIPTIONS, ConformalIntervals, RULModel, best_iteration, fit_model, imbalance_weights, make_model,
)

log = logging.getLogger(__name__)


def train_bundle(
    cfg, subset: str, train_raw: pd.DataFrame, test_raw: pd.DataFrame, test_rul: pd.DataFrame,
    version: str, tracker: Tracker, candidates: list[str] | None = None, fast: bool = False,
    data_sources: list[str] | None = None,
) -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    """Train everything for one subset. Returns (bundle, metadata, evaluation tables)."""
    t0 = time.time()
    seed = cfg["seed"]
    cap = cfg["data"]["rul_cap"]
    candidates = candidates or cfg["training"]["candidates"]

    # ---------------------------------------------------------------- split
    train_ids, val_ids = split_engines(train_raw["engine_id"].unique(), cfg["data"]["validation_fraction"], seed)
    labelled = add_train_rul(train_raw, cap)
    tr_raw = labelled[labelled["engine_id"].isin(train_ids)]
    va_raw = labelled[labelled["engine_id"].isin(val_ids)]

    # ------------------------------------------------------------- features
    fp = FeaturePipeline.from_config(cfg, subset).fit(tr_raw)
    tr, quality = fp.transform(tr_raw, return_report=True)
    va = fp.transform(va_raw)
    feats = fp.feature_names_
    X_tr, y_tr = tr[feats], tr["rul_capped"].to_numpy(float)
    X_va, y_va = va[feats], va["rul_capped"].to_numpy(float)
    w_tr = imbalance_weights(y_tr, cfg["training"]["imbalance_bins"], cfg["training"]["imbalance_power"])
    va_snap = truncation_snapshots(va, cfg["data"]["snapshots_per_val_engine"], seed)

    test_lab = add_test_rul(test_raw, test_rul, cap)
    te_all = fp.transform(test_lab)
    te_last = last_rows(te_all)

    experiment = f"fleetguard-{subset}"
    comparison, fitted = [], {}
    with tracker.run(experiment, f"{subset}-{version}", tags={"subset": subset, "version": version}) as parent:
        tracker.log_params({
            "subset": subset, "version": version, "train_engines": len(train_ids), "val_engines": len(val_ids),
            "train_rows": len(tr), "n_features": len(feats), "rul_cap": cap,
            "imbalance_power": cfg["training"]["imbalance_power"], "windows": fp.windows,
            "n_conditions": fp.n_conditions, "sensors": ",".join(fp.sensors_), "data_sources": data_sources,
        })

        # ------------------------------------------------- candidate models
        for name in candidates:
            t_start = time.time()
            with tracker.run(experiment, f"{subset}-{version}-{name}", nested=True, tags={"model": name}):
                model = fit_model(name, make_model(name, seed, fast), X_tr, y_tr, w_tr, X_va, y_va)
                rm = RULModel(name, model, feats, cap)
                m = regression_metrics(y_va, rm.predict_point(va), prefix="val_")
                m.update(regression_metrics(va_snap["rul"].clip(upper=cap), rm.predict_point(va_snap), prefix="val_snapshot_"))
                m.update(regression_metrics(te_last["rul"].clip(upper=cap), rm.predict_point(te_last), prefix="test_"))
                m["fit_seconds"] = time.time() - t_start
                tracker.log_params({"model": name, "best_iteration": best_iteration(name, model)})
                tracker.log_metrics(m)
            fitted[name] = rm
            comparison.append({"model": name, "description": MODEL_DESCRIPTIONS[name],
                               "best_iteration": best_iteration(name, model), **m})
            log.info("  %-14s val_rmse=%.2f test_rmse=%.2f (%.0fs)", name, m["val_rmse"], m["test_rmse"], m["fit_seconds"])

        comparison_df = pd.DataFrame(comparison).sort_values(cfg["training"]["selection_metric"]).reset_index(drop=True)
        best = comparison_df.iloc[0]["model"]
        rul_model = fitted[best]

        # ------------------------------------------------- anomaly detection
        a_cfg = cfg["anomaly"]
        detector = AnomalyDetector(fp.health_columns, a_cfg["threshold_quantile"], a_cfg["persistence"], seed)
        detector.fit(tr[tr["rul"] > a_cfg["healthy_rul"]], va[va["rul"] > a_cfg["healthy_rul"]])
        anomaly_val = detector.evaluate(va, a_cfg["degraded_rul"], a_cfg["healthy_rul"])

        # ---------------------------------------------- uncertainty intervals
        # Cross-conformal: residuals from out-of-fold predictions on the training
        # engines (refits of the chosen model) plus the validation snapshots. The
        # validation engines alone are too few, and were also used for early stopping.
        oof = out_of_fold(best, rul_model.estimator, tr, feats, y_tr, w_tr, seed, fast)
        tr_anom = detector.score(tr)["anomaly_index"].to_numpy()
        oof_snap = truncation_snapshots(tr.assign(_oof=oof, _anom=tr_anom), 10, seed)
        va_snap = va_snap.merge(
            va[["engine_id", "cycle"]].assign(_anom=detector.score(va)["anomaly_index"].to_numpy()),
            on=["engine_id", "cycle"], how="left",
        )
        calib_pred = np.r_[oof_snap["_oof"].to_numpy(), rul_model.predict_point(va_snap)]
        calib_true = np.r_[oof_snap["rul_capped"].to_numpy(), va_snap["rul"].clip(upper=cap).to_numpy()]
        calib_anom = np.r_[oof_snap["_anom"].to_numpy(), va_snap["_anom"].to_numpy()]
        rul_model.intervals = ConformalIntervals(cfg["training"]["interval_coverage"]).fit(calib_pred, calib_true)

        # ------------------------------------------------- risk assessment
        r_cfg = cfg.get("risk", {})
        risk = FailureRisk(r_cfg.get("horizon", 30), cap, r_cfg.get("health_rul_weight", 0.75))
        risk.fit(calib_pred, calib_anom, calib_true)

        # ---------------------------------------------- triage calibration
        t_cfg = cfg["triage"]
        calibration = None
        high_rul = t_cfg["high_rul"]
        if t_cfg["calibrate"]:
            calibration = calibrate_high_threshold(
                rul_model.predict_point(va_snap), va_snap["rul"], a_cfg["degraded_rul"],
                t_cfg["target_recall"], t_cfg["high_rul_bounds"],
            )
            high_rul = calibration["high_rul"]
        thresholds = TriageThresholds.from_config(cfg, high_rul=high_rul)

        # ------------------------------------------------ test evaluation
        te_pred = rul_model.predict(te_last)
        # Primary metrics compare against the true RUL capped like the training target
        # (a 150 vs 190 cycle difference is irrelevant for maintenance); uncapped kept for reference.
        test_m = regression_metrics(te_last["rul"].clip(upper=cap), te_pred["rul_pred"], prefix="test_")
        test_m.update(regression_metrics(te_last["rul"], te_pred["rul_pred"], prefix="test_uncapped_"))
        test_m.update(interval_metrics(te_last["rul"].clip(upper=cap), te_pred["rul_lower"], te_pred["rul_upper"], prefix="test_"))
        te_scores = detector.score(te_all)
        test_m.update(warning_metrics(te_all["rul"], te_scores["anomaly_index"], te_scores["anomaly_alarm"],
                                      a_cfg["degraded_rul"], prefix="test_anomaly_"))

        last_scores = te_scores.loc[te_all.groupby("engine_id")["cycle"].idxmax()].reset_index(drop=True)
        te_pred["failure_prob"] = risk.predict_proba(te_pred["rul_pred"], last_scores["anomaly_index"])
        te_pred["health_score"] = risk.health(te_pred["rul_pred"], last_scores["anomaly_index"])
        test_m.update(risk_metrics(te_pred["failure_prob"], te_pred["health_score"], te_last["rul"],
                                   risk.horizon, cap, prefix="test_"))

        agent = TriageAgent(thresholds)
        decisions = []
        for i, row in te_last.iterrows():
            state = {
                "engine_id": row["engine_id"], "cycle": row["cycle"], "n_cycles": int(row["cycle"]),
                "imputed_fraction": 0.0, **te_pred.loc[i].to_dict(), **last_scores.loc[i].to_dict(),
            }
            decisions.append(agent.decide(state, detector.sensor_deviation(row)).to_dict())
        decisions_df = pd.DataFrame(decisions)
        test_m.update(agent_metrics(decisions_df, te_last["rul"].values, a_cfg["degraded_rul"]))

        metrics = {
            **{k: v for k, v in comparison_df.iloc[0].items() if k.startswith("val")},
            **test_m,
            **{f"anomaly_{k}": v for k, v in anomaly_val.items() if not isinstance(v, str)},
            "triage_high_rul": high_rul,
            "train_seconds": time.time() - t0,
        }
        tracker.log_params({"selected_model": best, "selected_detector": detector.method,
                            "triage_high_rul": high_rul})
        tracker.log_metrics(metrics)

        # ------------------------------------------------- evaluation tables
        test_predictions = te_last[["engine_id", "cycle", "rul"]].rename(columns={"rul": "true_rul"}).join(te_pred)
        test_predictions = test_predictions.merge(
            decisions_df[["engine_id", "severity", "anomaly_index", "anomaly_alarm"]], on="engine_id"
        )
        tables = {
            "model_comparison": comparison_df,
            "test_predictions": test_predictions,
            "error_buckets": error_by_bucket(te_last["rul"], te_pred["rul_pred"]),
            "failure_prob_calibration": calibration_table(te_pred["failure_prob"], te_last["rul"] <= risk.horizon),
            "feature_importance": rul_model.feature_importance(30),
        }
        mlflow_version = tracker.log_model(rul_model.estimator, subset, X_va)
        _log_plots(tracker, tables, subset, version)
        run_id = parent.info.run_id if parent is not None else None

    bundle = {
        "subset": subset,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": fp,
        "model": rul_model,
        "anomaly": detector,
        "risk": risk,
        "thresholds": thresholds,
        "reference": reference_sample(tr, fp.health_columns + ["cycle"], seed=seed),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_version": mlflow_version,
    }
    metadata = {
        "subset": subset,
        "version": version,
        "model_name": best,
        "selected_detector": detector.method,
        "trained_engines": int(len(train_ids)),
        "validation_engines": int(len(val_ids)),
        "data_sources": data_sources or [],
        "feature_summary": fp.summary(),
        "data_quality": quality.to_dict(),
        "conformal": rul_model.intervals.to_dict(),
        "anomaly_thresholds": detector.thresholds_,
        "risk_model": risk.to_dict(),
        "triage_thresholds": thresholds.__dict__,
        "triage_calibration": calibration,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "mlflow_run_id": run_id,
        "mlflow_model_version": mlflow_version,
    }
    return bundle, metadata, tables


def out_of_fold(name, estimator, tr: pd.DataFrame, feats, y, w, seed: int, fast: bool, folds: int = 5) -> np.ndarray:
    """Out-of-fold predictions (GroupKFold by engine) with the selected model's settings.

    Boosted models are refitted with the number of trees early stopping chose,
    so no validation set is needed inside the folds.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    template = clone(estimator)
    if name == "xgboost":
        template.set_params(early_stopping_rounds=None, n_estimators=(best_iteration(name, estimator) or 300) + 1)
    elif name == "lightgbm":
        template.set_params(n_estimators=best_iteration(name, estimator) or 300)
    oof = np.zeros(len(tr))
    X = tr[feats]
    for a, c in GroupKFold(folds).split(X, groups=tr["engine_id"]):
        m = clone(template)
        if name == "ridge":
            m.fit(X.iloc[a], y[a], ridge__sample_weight=w[a])
        else:
            m.fit(X.iloc[a], y[a], sample_weight=w[a])
        oof[c] = m.predict(X.iloc[c])
    return np.clip(oof, 0, y.max())  # same clipping as RULModel.predict_point


def _log_plots(tracker: Tracker, tables: dict, subset: str, version: str) -> None:
    if not tracker.enabled:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        tp = tables["test_predictions"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(tp["true_rul"], tp["rul_pred"],
                    yerr=[tp["rul_pred"] - tp["rul_lower"], tp["rul_upper"] - tp["rul_pred"]],
                    fmt="o", ms=4, alpha=0.6, elinewidth=0.6)
        lim = max(tp["true_rul"].max(), tp["rul_upper"].max()) + 5
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set(xlabel="True RUL (cycles)", ylabel="Predicted RUL", title=f"{subset} {version}: test set")
        fig.tight_layout()
        fig.savefig(Path(tmp) / "test_pred_vs_true.png", dpi=110)
        plt.close(fig)

        fi = tables["feature_importance"].head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(fi["feature"], fi["importance"])
        ax.set(title="Top features", xlabel="Relative importance")
        fig.tight_layout()
        fig.savefig(Path(tmp) / "feature_importance.png", dpi=110)
        plt.close(fig)

        for name, df in tables.items():
            df.to_csv(Path(tmp) / f"{name}.csv", index=False)
        for p in Path(tmp).iterdir():
            tracker.log_artifact(p, "evaluation")


def save_tables(tables: dict[str, pd.DataFrame], out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
