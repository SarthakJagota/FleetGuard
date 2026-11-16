"""Predicting on new data: uploaded histories, single new readings, what-if scenarios.

The model needs an engine's *history* (rolling windows, slopes, drift from its
own baseline), so every entry point here works on a cycle-by-cycle history and
predicts for its latest cycle. A lone reading is handled honestly: the agent's
INSUFFICIENT_HISTORY rule flags it instead of returning a confident number.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .agent.work_orders import WorkOrderStore
from .data.loader import RAW_COLS, SENSOR_COLS, SETTING_COLS
from .scoring import score_fleet, score_history

# Accepted header aliases -> canonical names (people export data in many ways).
_ALIASES = {"unit": "engine_id", "unit_number": "engine_id", "engine": "engine_id", "id": "engine_id",
            "time": "cycle", "time_in_cycles": "cycle", "cycles": "cycle"}
for i in range(1, 4):
    for a in (f"op_setting_{i}", f"setting{i}", f"os{i}", f"operational_setting_{i}"):
        _ALIASES[a] = f"setting_{i}"
for i in range(1, 22):
    for a in (f"sensor{i}", f"s{i}", f"sensor_measurement_{i}", f"s_{i}"):
        _ALIASES[a] = f"sensor_{i}"


def read_upload(file_or_bytes, name: str = "upload.csv") -> pd.DataFrame:
    """Read an uploaded CSV (with headers) or a NASA-format .txt (26 unnamed columns)."""
    raw = file_or_bytes.getvalue() if hasattr(file_or_bytes, "getvalue") else file_or_bytes
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    if name.lower().endswith(".txt"):
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None).dropna(axis=1, how="all")
        if df.shape[1] != len(RAW_COLS):
            raise ValueError(f"NASA-format file must have {len(RAW_COLS)} columns, found {df.shape[1]}")
        df.columns = RAW_COLS
        return df
    df = pd.read_csv(io.StringIO(text))
    df.columns = [_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns.astype(str)]
    if "engine_id" not in df.columns and "cycle" in df.columns:
        df["engine_id"] = 1  # a single engine without an id column
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}. Download the template to see the expected format.")
    return df[RAW_COLS]


MULTI_REGIME = ("FD004", "FD002")   # preference order: two fault modes first (more general)
SINGLE_REGIME = ("FD003", "FD001")


def count_regimes(df: pd.DataFrame) -> int:
    """Distinct operating regimes (C-MAPSS settings sit on a few exact set-points)."""
    return int(len(df[SETTING_COLS].round(0).drop_duplicates()))


def is_multi_regime(df: pd.DataFrame) -> bool:
    return count_regimes(df) > 2


def suggest_subset(df: pd.DataFrame, available=None) -> tuple[str, str]:
    """Pick the model that fits the data: count operating regimes, prefer the
    model trained on both fault modes, and fall back to what is actually trained."""
    n = count_regimes(df)
    multi = n > 2
    preferred = MULTI_REGIME if multi else SINGLE_REGIME
    kind = (f"{n} operating regimes detected -> multi-regime model" if multi
            else "a single operating regime detected -> single-regime model")
    available = list(available) if available is not None else list(MULTI_REGIME + SINGLE_REGIME)
    for s in preferred:
        if s in available:
            return s, f"{kind} ({s})"
    fallback = available[0]
    return fallback, f"{kind}, but none is trained; using {fallback} (results may be unreliable)"


def template(n_cycles: int = 3) -> pd.DataFrame:
    """Empty-looking template with the exact columns (values are placeholders)."""
    df = pd.DataFrame(0.0, index=range(n_cycles), columns=RAW_COLS)
    df["engine_id"], df["cycle"] = 1, range(1, n_cycles + 1)
    return df


def predict_upload(bundle: dict, raw: pd.DataFrame, store: WorkOrderStore | None = None, fleet_label: str = "UPLOAD"):
    """Score uploaded engine histories. Returns (fleet table, row history, quality report)."""
    return score_fleet(bundle, raw, store, fleet_label=fleet_label)


def append_reading(history: pd.DataFrame, reading: dict) -> pd.DataFrame:
    """Add one new cycle to an engine's history (cycle = last + 1 unless given)."""
    row = {c: history[c].iloc[-1] for c in RAW_COLS}
    row.update({k: float(v) for k, v in reading.items() if k in SETTING_COLS + SENSOR_COLS})
    row["cycle"] = int(reading.get("cycle", history["cycle"].max() + 1))
    row["engine_id"] = int(history["engine_id"].iloc[0])
    return pd.concat([history[RAW_COLS], pd.DataFrame([row])], ignore_index=True)


def latest_prediction(bundle: dict, history: pd.DataFrame) -> dict:
    """Prediction + agent decision for the latest cycle of one engine."""
    fleet, hist, report = score_fleet(bundle, history)
    return {"decision": fleet.iloc[0].to_dict(), "history": hist, "quality": report}


def what_if(bundle: dict, history: pd.DataFrame, shifts_sd: dict[str, float], last_n: int) -> pd.DataFrame:
    """Return a copy of the history with sensors drifting by `shifts_sd` standard deviations.

    The shift ramps up linearly over the last `last_n` cycles (wear is gradual),
    in units of each sensor's spread *within its operating regime*, so +2 SD
    means the same severity for every sensor and every regime.
    """
    fp = bundle["features"]
    out = history[RAW_COLS].copy().reset_index(drop=True)
    cond = fp._conditions(out)
    n = len(out)
    k = max(1, min(last_n, n))
    ramp = np.zeros(n)
    ramp[n - k:] = np.linspace(1.0 / k, 1.0, k)
    for sensor, sd in shifts_sd.items():
        if not sd:
            continue
        std = fp.cond_std_[sensor].reindex(cond).to_numpy()
        std = np.where(np.isfinite(std), std, 0.0)
        out[sensor] = out[sensor] + sd * ramp * std
    return out


def compare(bundle: dict, base: pd.DataFrame, modified: pd.DataFrame) -> dict:
    """Score original and modified histories side by side (for the what-if view)."""
    b_fleet, b_hist, _ = score_fleet(bundle, base)
    m_fleet, m_hist, _ = score_fleet(bundle, modified)
    return {"before": b_fleet.iloc[0].to_dict(), "after": m_fleet.iloc[0].to_dict(),
            "before_hist": b_hist, "after_hist": m_hist}


__all__ = ["read_upload", "suggest_subset", "template", "predict_upload", "append_reading",
           "latest_prediction", "what_if", "compare", "score_history"]
