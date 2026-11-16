"""Risk assessment: failure probability and a 0-100 health score.

Failure probability
    P(engine fails within `horizon` cycles | predicted RUL, anomaly index).
    A logistic regression on two signals the system already produces, fitted
    on held-out calibration snapshots (out-of-fold training predictions +
    validation engines), so the probability is *calibrated*: of all engines
    given 30%, roughly 30% should really fail within the horizon. This is
    checked on the test set with the Brier score and a reliability curve.

Health score
    A transparent 0-100 index for dashboards and managers:
        100 * RUL_component * ( w + (1 - w) * anomaly_component )
    RUL_component     = predicted RUL / cap            (1 = fully healthy plateau)
    anomaly_component = 1 at anomaly index <= 1, falling linearly to 0 at index 3
    The anomaly part can only *lower* the score (by up to 1 - w = 25%), never lift
    it: an engine about to fail scores ~0 whatever its anomaly index. (A first,
    additive version gave near-failure engines a floor of 25 points.)
    Deliberately a formula, not a model, so anyone can recompute it by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

HEALTH_BANDS = [(80, "Good"), (50, "Fair"), (25, "Poor"), (0, "Critical")]


def health_score(rul_pred, anomaly_index, cap: float = 125, rul_weight: float = 0.75) -> np.ndarray:
    rul_c = np.clip(np.asarray(rul_pred, float) / cap, 0, 1)
    anom = np.nan_to_num(np.asarray(anomaly_index, float), nan=0.0)
    anom_c = np.clip(1 - (anom - 1) / 2, 0, 1)
    return np.round(100 * rul_c * (rul_weight + (1 - rul_weight) * anom_c), 1)


def health_band(score: float) -> str:
    for limit, name in HEALTH_BANDS:
        if score >= limit:
            return name
    return "Critical"


class FailureRisk:
    def __init__(self, horizon: int = 30, cap: float = 125, rul_weight: float = 0.75):
        self.horizon = horizon
        self.cap = cap
        self.rul_weight = rul_weight

    def _X(self, rul_pred, anomaly_index) -> np.ndarray:
        r = np.clip(np.asarray(rul_pred, float), 0, self.cap)
        a = np.clip(np.nan_to_num(np.asarray(anomaly_index, float), nan=0.0), 0, 5)
        return np.column_stack([r / self.cap, np.log1p(r), a])

    def fit(self, rul_pred, anomaly_index, true_rul) -> "FailureRisk":
        y = (np.asarray(true_rul, float) <= self.horizon).astype(int)
        if y.min() == y.max():
            raise ValueError("Calibration data needs both failing and healthy examples")
        self.clf_ = LogisticRegression(C=1.0, max_iter=1000).fit(self._X(rul_pred, anomaly_index), y)
        self.base_rate_ = float(y.mean())
        return self

    def predict_proba(self, rul_pred, anomaly_index) -> np.ndarray:
        return self.clf_.predict_proba(self._X(rul_pred, anomaly_index))[:, 1]

    def health(self, rul_pred, anomaly_index) -> np.ndarray:
        return health_score(rul_pred, anomaly_index, self.cap, self.rul_weight)

    def to_dict(self) -> dict:
        names = ["rul/cap", "log1p(rul)", "anomaly_index"]
        return {
            "horizon": self.horizon,
            "calibration_base_rate": round(self.base_rate_, 4),
            "coefficients": {n: round(float(c), 4) for n, c in zip(names, self.clf_.coef_[0])},
            "intercept": round(float(self.clf_.intercept_[0]), 4),
            "health_rul_weight": self.rul_weight,
        }


def calibration_table(prob, outcome, bins: int = 10) -> pd.DataFrame:
    """Reliability curve: mean predicted probability vs observed frequency per bin."""
    df = pd.DataFrame({"p": np.asarray(prob, float), "y": np.asarray(outcome, int)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return out.reset_index(drop=True)


def risk_metrics(prob, health, true_rul, horizon: int, cap: float, prefix: str = "") -> dict:
    true_rul = np.asarray(true_rul, float)
    y = (true_rul <= horizon).astype(int)
    out = {f"{prefix}failure_prob_brier": float(brier_score_loss(y, prob)),
           f"{prefix}failure_prob_base_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        out[f"{prefix}failure_prob_auc"] = float(roc_auc_score(y, prob))
    cal = calibration_table(prob, y)
    # Expected calibration error: bin-size-weighted gap between predicted and observed.
    out[f"{prefix}failure_prob_ece"] = float((cal["n"] * (cal["predicted"] - cal["observed"]).abs()).sum() / cal["n"].sum())
    from scipy.stats import spearmanr

    out[f"{prefix}health_spearman_rul"] = float(spearmanr(health, np.minimum(true_rul, cap)).statistic)
    return out
