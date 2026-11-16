"""Remaining-Useful-Life regressors, imbalance weighting and conformal intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DESCRIPTIONS = {
    "ridge": "Ridge regression - linear, fully explainable baseline",
    "random_forest": "Random forest - non-linear bagging, robust to noisy features",
    "xgboost": "XGBoost - gradient-boosted trees with early stopping",
    "lightgbm": "LightGBM - histogram gradient boosting with early stopping",
}


def make_model(name: str, seed: int = 42, fast: bool = False):
    """Factory for candidate models. ``fast`` shrinks them for unit tests."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=20 if fast else 200, max_depth=16, min_samples_leaf=4,
            max_features=0.33, n_jobs=-1, random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=50 if fast else 2000, learning_rate=0.03, max_depth=5,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            tree_method="hist", early_stopping_rounds=None if fast else 100,
            n_jobs=-1, random_state=seed, verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=50 if fast else 2000, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
    raise ValueError(f"Unknown model {name!r}")


def fit_model(name: str, model, X, y, w, X_val, y_val):
    """Fit with early stopping on the validation engines where supported."""
    if name == "xgboost" and getattr(model, "early_stopping_rounds", None):
        model.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm" and model.n_estimators > 100:
        import inspect

        from lightgbm import early_stopping, log_evaluation

        # LightGBM >= 4.7 renamed eval_set to eval_X / eval_y.
        if "eval_X" in inspect.signature(model.fit).parameters:
            evals = dict(eval_X=(X_val,), eval_y=(y_val,))
        else:
            evals = dict(eval_set=[(X_val, y_val)])
        model.fit(X, y, sample_weight=w, callbacks=[early_stopping(100, verbose=False), log_evaluation(0)], **evals)
    elif name == "ridge":
        model.fit(X, y, ridge__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def best_iteration(name: str, model) -> int | None:
    if name == "xgboost":
        return getattr(model, "best_iteration", None)
    if name == "lightgbm":
        return getattr(model, "best_iteration_", None)
    return None


def imbalance_weights(y: np.ndarray, bins, power: float) -> np.ndarray:
    """Up-weight the rare low-RUL rows.

    With a 125 cap most rows sit on the healthy plateau and only a few are
    near failure - exactly the rows that matter for maintenance. Weight =
    (inverse bin frequency) ** power, normalised to mean 1. power=0.5 is a
    deliberately mild correction (the brief asks for 'mild imbalance').
    """
    if power <= 0:
        return np.ones(len(y))
    idx = np.clip(np.digitize(y, bins) - 1, 0, len(bins) - 2)
    counts = np.bincount(idx, minlength=len(bins) - 1).astype(float)
    freq = counts[idx] / len(y)
    w = (1.0 / np.maximum(freq, 1e-9)) ** power
    return w / w.mean()


class ConformalIntervals:
    """Split-conformal RUL intervals, calibrated per predicted-RUL band.

    Errors grow with the RUL horizon, so one global width would be too wide
    near failure and too narrow far away. We take quantiles of the signed
    residual (true - predicted) on held-out validation engines within each
    band of the prediction (a 'Mondrian' conformal predictor).
    """

    def __init__(self, coverage: float = 0.8, edges=(0, 30, 60, 90, np.inf), min_points: int = 30):
        self.coverage = coverage
        self.edges = np.asarray(edges, float)
        self.min_points = min_points

    def fit(self, y_pred, y_true) -> "ConformalIntervals":
        y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
        resid = y_true - y_pred
        lo_q, hi_q = (1 - self.coverage) / 2, (1 + self.coverage) / 2
        glob = (float(np.quantile(resid, lo_q)), float(np.quantile(resid, hi_q)))
        self.bands_ = []
        band = np.clip(np.digitize(y_pred, self.edges) - 1, 0, len(self.edges) - 2)
        for b in range(len(self.edges) - 1):
            r = resid[band == b]
            if len(r) >= self.min_points:
                self.bands_.append((float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q))))
            else:
                self.bands_.append(glob)
        return self

    def predict(self, y_pred) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, float)
        band = np.clip(np.digitize(y_pred, self.edges) - 1, 0, len(self.edges) - 2)
        lo = np.array([self.bands_[b][0] for b in band])
        hi = np.array([self.bands_[b][1] for b in band])
        return np.clip(y_pred + lo, 0, None), np.maximum(y_pred + hi, y_pred)

    def to_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "bands": [
                {"pred_from": float(a), "pred_to": float(b), "resid_lo": round(lo, 2), "resid_hi": round(hi, 2)}
                for (a, b), (lo, hi) in zip(zip(self.edges[:-1], self.edges[1:]), self.bands_)
            ],
        }


class RULModel:
    """Fitted estimator + its feature order, output clipping and intervals."""

    def __init__(self, name: str, estimator, feature_names: list[str], cap: int):
        self.name = name
        self.estimator = estimator
        self.feature_names = list(feature_names)
        self.cap = cap
        self.intervals: ConformalIntervals | None = None

    def predict_point(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_names]
        return np.clip(self.estimator.predict(X), 0, self.cap)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict_point(features)
        out = pd.DataFrame({"rul_pred": pred}, index=features.index)
        if self.intervals is not None:
            lo, hi = self.intervals.predict(pred)
            out["rul_lower"], out["rul_upper"] = lo, hi
        return out

    def feature_importance(self, top: int = 25) -> pd.DataFrame:
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, float)
        elif hasattr(est, "named_steps"):
            imp = np.abs(est.named_steps["ridge"].coef_)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = imp / max(imp.sum(), 1e-12)
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=False).head(top).reset_index(drop=True)
        )
