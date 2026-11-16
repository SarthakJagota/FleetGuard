"""Evaluation metrics for RUL regression, early warning and the triage agent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (over-estimating RUL) cost more.

    d = pred - true; s = exp(-d/13) - 1 if d < 0 else exp(d/10) - 1. Lower is better.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "nasa_score": nasa_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "late_pct": float(np.mean(y_pred > y_true + 10) * 100),  # dangerous over-estimates
    }
    return {f"{prefix}{k}": v for k, v in m.items()}


def interval_metrics(y_true, lower, upper, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        f"{prefix}interval_coverage": float(np.mean(covered)),
        f"{prefix}interval_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
    }


def warning_metrics(true_rul, score, flag, horizon: int, prefix: str = "") -> dict[str, float]:
    """Binary 'fails within `horizon` cycles' detection. Classes are imbalanced,
    so we report PR-AUC / precision / recall / F1 instead of accuracy."""
    y = (np.asarray(true_rul) <= horizon).astype(int)
    out = {f"{prefix}positive_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        out[f"{prefix}roc_auc"] = float(roc_auc_score(y, score))
        out[f"{prefix}pr_auc"] = float(average_precision_score(y, score))
    out[f"{prefix}precision"] = float(precision_score(y, flag, zero_division=0))
    out[f"{prefix}recall"] = float(recall_score(y, flag, zero_division=0))
    out[f"{prefix}f1"] = float(f1_score(y, flag, zero_division=0))
    return out


def error_by_bucket(y_true, y_pred, edges=(0, 25, 50, 75, 100, 1000)) -> pd.DataFrame:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    labels = [f"{a}-{b - 1}" if b < 1000 else f"{a}+" for a, b in zip(edges[:-1], edges[1:])]
    df["bucket"] = pd.cut(df["true"], bins=list(edges), labels=labels, right=False)
    df["err"] = df["pred"] - df["true"]
    return (
        df.groupby("bucket", observed=True)
        .agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"))
        .reset_index()
    )


def agent_metrics(decisions: pd.DataFrame, true_rul: pd.Series, degraded: int = 30, healthy: int = 90) -> dict:
    """How well does the agent turn predictions into the right actions?

    * recall: engines truly within `degraded` cycles of failure that got a work order
    * false-alarm rate: work orders raised on engines with > `healthy` cycles left
    * missed-critical: engines within 15 cycles that were left on LOW/MEDIUM
    """
    d = decisions.copy()
    d["true_rul"] = np.asarray(true_rul)
    wo = d["severity"].isin(["CRITICAL", "HIGH"])
    review = d["severity"] == "REVIEW"
    near = d["true_rul"] <= degraded
    safe = d["true_rul"] > healthy
    crit = d["true_rul"] <= 15
    acted = wo | review  # a human reviewing it also counts as "not missed"
    return {
        "agent_recall_near_failure": float((acted & near).sum() / max(near.sum(), 1)),
        "agent_work_order_precision": float((wo & (d["true_rul"] <= 60)).sum() / max(wo.sum(), 1)),
        "agent_false_alarm_rate": float((wo & safe).sum() / max(safe.sum(), 1)),
        "agent_missed_critical": int((crit & ~acted).sum()),
        "agent_review_rate": float(review.mean()),
        "agent_work_orders": int(wo.sum()),
    }
