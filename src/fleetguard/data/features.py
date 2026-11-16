"""Feature engineering.

Steps (fit on training engines only, then applied unchanged everywhere):

1. Operating-condition clustering - FD002/FD004 fly six flight regimes and
   raw sensor values jump between them. KMeans on the 3 settings recovers
   the regime of every row.
2. Condition normalisation - z-score each sensor with the mean/std of its
   regime so a reading is comparable across regimes.
3. Sensor selection - drop sensors that are constant or carry no RUL signal.
4. Noise handling - spike replacement + exponential smoothing (EWMA).
5. Rolling-window features - mean, std and least-squares slope over short
   and long windows, plus drift from the engine's own early-life baseline.

Every feature at cycle t uses only cycles <= t (strictly causal), so the
prediction for a truncated history equals the prediction for that row of
the full history. This is what lets the same code serve batch scoring and
cycle-by-cycle streaming, and it is covered by a unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .cleaning import QualityReport, replace_outliers, validate_and_repair
from .loader import SENSOR_COLS, SETTING_COLS


class FeaturePipeline:
    def __init__(
        self,
        n_conditions: int = 1,
        windows: tuple[int, ...] = (10, 30),
        ewma_alpha: float = 0.3,
        baseline_cycles: int = 10,
        min_sensor_corr: float = 0.05,
        outlier_z: float = 6.0,
        max_gap_fill: int = 5,
        rul_cap: int = 125,
        seed: int = 42,
    ):
        self.n_conditions = n_conditions
        self.windows = tuple(windows)
        self.ewma_alpha = ewma_alpha
        self.baseline_cycles = baseline_cycles
        self.min_sensor_corr = min_sensor_corr
        self.outlier_z = outlier_z
        self.max_gap_fill = max_gap_fill
        self.rul_cap = rul_cap
        self.seed = seed
        self.fitted_ = False

    @classmethod
    def from_config(cls, cfg, subset: str) -> "FeaturePipeline":
        f, d = cfg["features"], cfg["data"]
        return cls(
            n_conditions=cfg.subset(subset)["operating_conditions"],
            windows=tuple(f["windows"]),
            ewma_alpha=f["ewma_alpha"],
            baseline_cycles=f["baseline_cycles"],
            min_sensor_corr=f["min_sensor_corr"],
            outlier_z=d["outlier_z"],
            max_gap_fill=d["max_gap_fill"],
            rul_cap=d["rul_cap"],
            seed=cfg["seed"],
        )

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        """Learn regimes, per-regime statistics and the useful sensor set."""
        clean, _ = validate_and_repair(train, self.max_gap_fill)
        self.scaler_ = StandardScaler().fit(clean[SETTING_COLS])
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans_.fit(self.scaler_.transform(clean[SETTING_COLS]))
        else:
            self.kmeans_ = None
        clean["condition"] = self._conditions(clean)

        grouped = clean.groupby("condition")[SENSOR_COLS]
        self.cond_mean_ = grouped.mean()
        std = grouped.std(ddof=0)
        # Relative floor: a sensor whose spread is tiny compared with its level is constant.
        floor = 1e-6 * self.cond_mean_.abs().clip(lower=1.0)
        self.cond_std_ = std.where(std > floor, np.nan)

        # Keep sensors that vary within at least one regime and track RUL.
        z = self._normalise(clean)
        rul = (clean.groupby("engine_id")["cycle"].transform("max") - clean["cycle"]).clip(upper=self.rul_cap)
        self.sensor_corr_ = {}
        selected = []
        for s in SENSOR_COLS:
            col = z[s]
            if col.notna().sum() < 10 or col.std() < 1e-6:
                self.sensor_corr_[s] = 0.0
                continue
            rho = spearmanr(col, rul, nan_policy="omit").statistic
            rho = 0.0 if not np.isfinite(rho) else float(rho)
            self.sensor_corr_[s] = rho
            if abs(rho) >= self.min_sensor_corr:
                selected.append(s)
        if not selected:
            raise ValueError("No informative sensors found during feature fitting")
        self.sensors_ = selected

        self.fitted_ = True
        self.feature_names_ = list(self._build(clean.head(200)).columns)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, return_report: bool = False):
        """Clean + featurise. Returns the frame of row metadata + features."""
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline must be fitted first")
        clean, report = validate_and_repair(df, self.max_gap_fill)
        clean["condition"] = self._conditions(clean)
        z = self._normalise(clean)[self.sensors_]
        clean = replace_outliers(clean, z, self.outlier_z, report)
        feats = self._build(clean)
        meta_cols = ["engine_id", "cycle", "condition", "imputed"] + [
            c for c in ("rul", "rul_capped") if c in clean.columns
        ]
        # `cycle` is both metadata and a model feature; keep a single column.
        feats = feats.drop(columns="cycle").reset_index(drop=True)
        out = pd.concat([clean[meta_cols].reset_index(drop=True), feats], axis=1)
        return (out, report) if return_report else out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------- helpers
    def _conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(self.scaler_.transform(df[SETTING_COLS]))

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        cond = df["condition"].to_numpy()
        mean = self.cond_mean_.reindex(cond).to_numpy()
        std = self.cond_std_.reindex(cond).to_numpy()
        with np.errstate(invalid="ignore"):
            z = (df[SENSOR_COLS].to_numpy(dtype=float) - mean) / std
        z = pd.DataFrame(z, columns=SENSOR_COLS, index=df.index)
        # A sensor constant in this regime but not in others carries no information here.
        return z.fillna(0.0)

    def _build(self, clean: pd.DataFrame) -> pd.DataFrame:
        clean = clean.reset_index(drop=True)
        z = self._normalise(clean)[self.sensors_]
        z.columns = [f"{s}_z" for s in self.sensors_]
        engine = clean["engine_id"].to_numpy()
        frame = z.copy()
        frame["engine_id"] = engine
        g = frame.groupby("engine_id", sort=False)

        zcols = list(z.columns)
        ewm = g[zcols].ewm(alpha=self.ewma_alpha, adjust=True).mean().reset_index(level=0, drop=True).sort_index()
        ewm.columns = [c.replace("_z", "_ewm") for c in zcols]

        # Each engine's own early-life baseline: mean of its first N cycles (expanding until N).
        pos = g.cumcount().to_numpy()
        early = z.where(np.broadcast_to((pos < self.baseline_cycles)[:, None], z.shape))
        early["engine_id"] = engine
        base = (
            early.groupby("engine_id", sort=False)[zcols].expanding().mean()
            .reset_index(level=0, drop=True).sort_index()
        )  # expanding mean skips NaN, so after N cycles it stays at the first-N mean
        delta = pd.DataFrame(ewm.to_numpy() - base.to_numpy(), columns=[c.replace("_z", "_drift") for c in zcols])

        parts = [ewm, delta]
        x = clean["cycle"].to_numpy(dtype=float)
        for w in self.windows:
            parts.extend(self._rolling(z, engine, x, w))

        feats = pd.concat(parts, axis=1)
        feats["cycle"] = clean["cycle"].to_numpy()
        return feats.fillna(0.0)

    def _rolling(self, z: pd.DataFrame, engine: np.ndarray, x: np.ndarray, w: int) -> list[pd.DataFrame]:
        """Vectorised rolling mean, std and OLS slope (no per-window Python calls)."""
        names = [c.replace("_z", "") for c in z.columns]
        y = z.to_numpy()
        # Centre x per engine to keep the slope formula numerically stable.
        xs = x - pd.Series(x).groupby(engine).transform("min").to_numpy()
        data = np.hstack([y, y * xs[:, None], xs[:, None], (xs ** 2)[:, None], np.ones((len(xs), 1))])
        k = y.shape[1]
        sums = (
            pd.DataFrame(data).groupby(engine, sort=False)
            .rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True).sort_index().to_numpy()
        )
        sy, sxy, sx, sxx, n = sums[:, :k], sums[:, k:2 * k], sums[:, 2 * k], sums[:, 2 * k + 1], sums[:, 2 * k + 2]
        mean = sy / n[:, None]
        sq = (
            pd.DataFrame(y ** 2).groupby(engine, sort=False)
            .rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True).sort_index().to_numpy()
        )
        var = np.clip(sq / n[:, None] - mean ** 2, 0, None) * (n / np.maximum(n - 1, 1))[:, None]
        std = np.where(n[:, None] >= 2, np.sqrt(var), 0.0)
        denom = n * sxx - sx ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (n[:, None] * sxy - sx[:, None] * sy) / denom[:, None]
        slope = np.where((n[:, None] >= 3) & (denom[:, None] > 0), slope, 0.0)
        return [
            pd.DataFrame(mean, columns=[f"{s}_mean{w}" for s in names]),
            pd.DataFrame(std, columns=[f"{s}_std{w}" for s in names]),
            pd.DataFrame(slope, columns=[f"{s}_slope{w}" for s in names]),
        ]

    # ------------------------------------------------------------- columns
    @property
    def health_columns(self) -> list[str]:
        """Smoothed condition-normalised sensors: input to the anomaly detectors."""
        return [f"{s}_ewm" for s in self.sensors_]

    def summary(self) -> dict:
        return {
            "n_conditions": self.n_conditions,
            "sensors": self.sensors_,
            "dropped_sensors": [s for s in SENSOR_COLS if s not in self.sensors_],
            "sensor_spearman_rul": {k: round(v, 4) for k, v in self.sensor_corr_.items()},
            "n_features": len(self.feature_names_),
            "windows": list(self.windows),
        }


def last_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Latest observation per engine (what the fleet looks like 'now')."""
    idx = features.groupby("engine_id")["cycle"].idxmax()
    return features.loc[idx].reset_index(drop=True)


def truncation_snapshots(features: pd.DataFrame, per_engine: int, seed: int, min_cycles: int = 15) -> pd.DataFrame:
    """Random cut points per engine, mimicking how the official test set is built."""
    rng = np.random.default_rng(seed)
    picks = []
    for _, grp in features.groupby("engine_id"):
        eligible = grp.index[grp["cycle"] >= min(min_cycles, grp["cycle"].max())]
        k = min(per_engine, len(eligible))
        picks.extend(rng.choice(eligible, size=k, replace=False))
    return features.loc[sorted(picks)].reset_index(drop=True)
