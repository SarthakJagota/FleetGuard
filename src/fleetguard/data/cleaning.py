"""Data validation and repair.

C-MAPSS itself is clean, but a deployed fleet is not: sensors drop out,
messages are duplicated or arrive out of order, and single readings spike.
These functions make the pipeline robust to that and record what they
repaired so the triage agent can send heavily repaired engines to a human.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .loader import ID_COLS, SENSOR_COLS, SETTING_COLS

VALUE_COLS = SETTING_COLS + SENSOR_COLS


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    engines: int = 0
    dropped_bad_ids: int = 0
    dropped_duplicates: int = 0
    reordered_engines: int = 0
    missing_values: int = 0
    imputed_values: int = 0
    dropped_unrepairable: int = 0
    outliers_replaced: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def validate_and_repair(df: pd.DataFrame, max_gap_fill: int = 5) -> tuple[pd.DataFrame, QualityReport]:
    """Schema checks, de-duplication, ordering and gap filling.

    Returns a cleaned copy with an ``imputed`` column (number of values
    filled on that row) and a report of what was changed.
    """
    missing_cols = [c for c in ID_COLS + VALUE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    report = QualityReport(rows_in=len(df))
    out = df.copy()
    for col in ID_COLS + VALUE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[VALUE_COLS] = out[VALUE_COLS].replace([np.inf, -np.inf], np.nan)

    # Rows without a usable engine id / cycle cannot be placed in time.
    bad_ids = out["engine_id"].isna() | out["cycle"].isna() | (out["cycle"] <= 0)
    report.dropped_bad_ids = int(bad_ids.sum())
    out = out.loc[~bad_ids].copy()
    out["engine_id"] = out["engine_id"].astype(int)
    out["cycle"] = out["cycle"].astype(int)

    # Out-of-order arrival: sort; duplicates: keep the latest message.
    unordered = out.groupby("engine_id")["cycle"].apply(lambda c: not c.is_monotonic_increasing)
    report.reordered_engines = int(unordered.sum())
    before = len(out)
    out = out.drop_duplicates(subset=ID_COLS, keep="last")
    report.dropped_duplicates = before - len(out)
    out = out.sort_values(ID_COLS).reset_index(drop=True)

    # Missing values: interpolate short gaps inside each engine only.
    missing_mask = out[VALUE_COLS].isna()
    report.missing_values = int(missing_mask.to_numpy().sum())
    if report.missing_values:
        filled = (
            out.groupby("engine_id")[VALUE_COLS]
            .transform(lambda s: s.interpolate(limit=max_gap_fill, limit_direction="both"))
        )
        out[VALUE_COLS] = filled
    still_missing = out[VALUE_COLS].isna()
    report.imputed_values = report.missing_values - int(still_missing.to_numpy().sum())
    out["imputed"] = (missing_mask & ~still_missing).sum(axis=1).astype(int)

    unrepairable = still_missing.any(axis=1)
    report.dropped_unrepairable = int(unrepairable.sum())
    out = out.loc[~unrepairable].reset_index(drop=True)

    report.rows_out = len(out)
    report.engines = int(out["engine_id"].nunique())
    return out, report


def replace_outliers(
    df: pd.DataFrame, z: pd.DataFrame, threshold: float, report: QualityReport | None = None
) -> pd.DataFrame:
    """Replace isolated spikes (|z| > threshold) by interpolation within the engine.

    ``z`` holds condition-normalised sensor values aligned with ``df``.
    Degradation is gradual, so a single reading 6 sigma from its operating
    condition's mean is far more likely a glitch than real wear.
    """
    cols = list(z.columns)
    spikes = z.abs() > threshold
    n = int(spikes.to_numpy().sum())
    if report is not None:
        report.outliers_replaced += n
    if n == 0:
        return df
    out = df.copy()
    raw = out[cols].mask(spikes.values)
    raw["engine_id"] = out["engine_id"].values
    out[cols] = raw.groupby("engine_id")[cols].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    out["imputed"] = out["imputed"] + spikes.sum(axis=1).values
    return out


def inject_faults(
    df: pd.DataFrame, missing_rate: float = 0.02, spike_rate: float = 0.002,
    duplicate_rate: float = 0.01, seed: int = 0,
) -> pd.DataFrame:
    """Corrupt a clean dataframe the way a real telemetry feed would (for tests/demos)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    sensor_values = out[SENSOR_COLS].to_numpy(dtype=float)
    mask = rng.random(sensor_values.shape) < missing_rate
    sensor_values[mask] = np.nan
    spikes = rng.random(sensor_values.shape) < spike_rate
    std = np.nanstd(sensor_values, axis=0) + 1e-3
    sensor_values[spikes] += (rng.choice([-1, 1], size=spikes.sum()) * 15 * np.broadcast_to(std, sensor_values.shape)[spikes])
    out[SENSOR_COLS] = sensor_values
    dup_idx = rng.choice(len(out), size=int(len(out) * duplicate_rate), replace=False)
    out = pd.concat([out, out.iloc[dup_idx]], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # out-of-order arrival
