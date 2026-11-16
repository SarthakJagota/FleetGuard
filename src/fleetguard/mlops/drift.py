"""Data-drift monitoring with the Population Stability Index (PSI).

PSI = sum((cur% - ref%) * ln(cur% / ref%)) over quantile bins of the
reference distribution. Rule of thumb: < 0.1 stable, 0.1-0.2 moderate
shift, > 0.2 significant shift worth investigating / retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], alert: float = 0.2) -> pd.DataFrame:
    rows = []
    for col in columns:
        value = psi(reference[col], current[col])
        status = "stable" if value < 0.1 else "moderate" if value < alert else "drift"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def reference_sample(features: pd.DataFrame, columns: list[str], n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Small sample of training features kept in the model bundle for later drift checks."""
    sample = features[columns]
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    return sample.reset_index(drop=True)
