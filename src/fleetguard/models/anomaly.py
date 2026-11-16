"""Unsupervised degradation / anomaly detection.

All three detectors learn what a *healthy* engine looks like (early-life
rows, true RUL above the cap) and score how far each new reading is from that:

* Mahalanobis distance - statistical baseline, accounts for correlated sensors.
* Isolation Forest     - non-parametric, no distribution assumption.
* Autoencoder          - a small neural network (sensors -> 12 -> 4 -> 12 -> sensors)
                         trained to reconstruct healthy readings through a 4-unit
                         bottleneck; the reconstruction error is the score, so
                         patterns it never saw in healthy engines rebuild badly.
                         Built with scikit-learn's MLPRegressor to stay lightweight.

Alarm threshold = high quantile of scores on *healthy validation engines*,
i.e. we fix the false-alarm rate on healthy data rather than guessing a
contamination rate. A single exceedance is noise; an alarm needs
``persistence`` consecutive anomalous cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

from .metrics import warning_metrics


METHODS = ("mahalanobis", "iforest", "autoencoder")


class AnomalyDetector:
    def __init__(self, columns: list[str], threshold_quantile: float = 0.99, persistence: int = 3, seed: int = 42):
        self.columns = list(columns)
        self.threshold_quantile = threshold_quantile
        self.persistence = persistence
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, healthy: pd.DataFrame, healthy_val: pd.DataFrame) -> "AnomalyDetector":
        X = healthy[self.columns].to_numpy()
        self.center_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-9
        self.cov_ = LedoitWolf().fit(X)
        self.iforest_ = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(X)
        Xs = (X - self.center_) / self.scale_
        width = max(4, min(12, X.shape[1] - 2))
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=(width, 4, width), activation="tanh", alpha=1e-4, learning_rate_init=1e-3,
            max_iter=600, early_stopping=True, n_iter_no_change=15, random_state=self.seed,
        ).fit(Xs, Xs)
        self.thresholds_ = {
            m: float(np.quantile(self._raw(healthy_val, m), self.threshold_quantile))
            for m in METHODS
        }
        self.method = "mahalanobis"
        return self

    def _raw(self, df: pd.DataFrame, method: str) -> np.ndarray:
        X = df[self.columns].to_numpy()
        if method == "mahalanobis":
            return np.sqrt(self.cov_.mahalanobis(X))
        if method == "autoencoder":
            Xs = (X - self.center_) / self.scale_
            return np.mean((Xs - self.autoencoder_.predict(Xs)) ** 2, axis=1)
        return -self.iforest_.score_samples(X)

    # ---------------------------------------------------------------- score
    def score(self, df: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
        """Per-row anomaly index (score / threshold, >1 = anomalous) and persistent alarm."""
        method = method or self.method
        idx = self._raw(df, method) / self.thresholds_[method]
        out = pd.DataFrame({"anomaly_index": idx}, index=df.index)
        above = pd.Series(idx > 1.0, index=df.index)
        run = above.groupby(df["engine_id"].values).transform(
            lambda s: s.groupby((~s).cumsum()).cumsum()
        )
        out["anomaly_run"] = run.astype(int)
        out["anomaly_alarm"] = out["anomaly_run"] >= self.persistence
        return out

    def sensor_deviation(self, row: pd.Series) -> pd.Series:
        """Signed deviation of each smoothed sensor from the healthy centre (in healthy SDs)."""
        vals = row[self.columns].to_numpy(dtype=float)
        dev = (vals - self.center_) / self.scale_
        return pd.Series(dev, index=[c.replace("_ewm", "") for c in self.columns])

    # ------------------------------------------------------------- evaluate
    def evaluate(self, val_feats: pd.DataFrame, horizon: int, healthy_rul: int) -> dict:
        """Compare the detectors on validation engines and keep the best one (by PR-AUC)."""
        results = {}
        for method in METHODS:
            s = self.score(val_feats, method)
            m = warning_metrics(val_feats["rul"], s["anomaly_index"], s["anomaly_alarm"], horizon, prefix=f"{method}_")
            m.update(self._lead_time(val_feats, s, healthy_rul, prefix=f"{method}_"))
            results[method] = m
        self.method = max(results, key=lambda k: results[k].get(f"{k}_pr_auc", 0))
        flat = {k: v for r in results.values() for k, v in r.items()}
        flat["selected_detector"] = self.method
        return flat

    def _lead_time(self, feats: pd.DataFrame, scores: pd.DataFrame, healthy_rul: int, prefix: str) -> dict:
        """How many cycles before failure does the first persistent alarm fire?"""
        df = feats[["engine_id", "rul"]].join(scores)
        leads, early_false = [], 0
        for _, g in df.groupby("engine_id"):
            alarms = g[g["anomaly_alarm"]]
            if alarms.empty:
                continue
            first_rul = int(alarms["rul"].iloc[0])
            if first_rul > healthy_rul + 50:  # fired while clearly healthy
                early_false += 1
            leads.append(first_rul)
        n = df["engine_id"].nunique()
        return {
            f"{prefix}detected_engines_pct": 100.0 * len(leads) / max(n, 1),
            f"{prefix}median_lead_cycles": float(np.median(leads)) if leads else 0.0,
            f"{prefix}healthy_false_alarm_engines_pct": 100.0 * early_false / max(n, 1),
        }
