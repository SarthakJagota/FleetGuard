"""SHAP explanations of RUL predictions, summarised per sensor.

The model sees ~140 engineered features (rolling means, slopes, drift ...),
which are meaningless to a technician. For each prediction we compute SHAP
contributions (how much each feature moved the prediction away from the
average engine) and add them up per physical sensor, so a work order can say
"Ps30 lowered the predicted RUL by 21 cycles".

* Tree models (XGBoost, LightGBM, Random Forest) use shap.TreeExplainer (exact).
* The Ridge baseline is linear, so contribution = coefficient * standardised value.

SHAP explains the *model*, not the physics: it shows what drove the
prediction, not a proven root cause. Work orders say so.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .data.loader import SENSOR_INFO

log = logging.getLogger(__name__)
_SENSOR = re.compile(r"^(sensor_\d+)_")


def feature_sensor(feature: str) -> str:
    m = _SENSOR.match(feature)
    return m.group(1) if m else "cycle"


class RULExplainer:
    def __init__(self, rul_model):
        self.model = rul_model
        self.features = rul_model.feature_names
        self.sensors = np.array([feature_sensor(f) for f in self.features])
        est = rul_model.estimator
        self._linear = hasattr(est, "named_steps")
        if self._linear:
            scaler, ridge = est.named_steps["standardscaler"], est.named_steps["ridge"]
            self._mean, self._scale, self._coef = scaler.mean_, scaler.scale_, ridge.coef_
            self.base_value = float(ridge.intercept_)
        else:
            import shap

            self._tree = shap.TreeExplainer(est)
            ev = self._tree.expected_value
            self.base_value = float(np.ravel(ev)[0])

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values, shape (rows, features). Row sum + base_value = raw model output."""
        X = X[self.features]
        if self._linear:
            return (X.to_numpy(float) - self._mean) / self._scale * self._coef
        return np.asarray(self._tree.shap_values(X), float)

    def by_sensor(self, contrib: np.ndarray) -> pd.DataFrame:
        """Sum contributions over each sensor's features -> (rows, sensors)."""
        df = pd.DataFrame(np.atleast_2d(contrib), columns=self.sensors)
        return df.T.groupby(level=0).sum().T

    def drivers(self, X: pd.DataFrame, top: int = 3, min_cycles: float = 0.5) -> list[list[dict]]:
        """Per row: the sensors that lowered the predicted RUL the most
        (contributions smaller than `min_cycles` are noise and are left out)."""
        if len(X) == 0:
            return []
        per_sensor = self.by_sensor(self.contributions(X))
        out = []
        for _, row in per_sensor.iterrows():
            neg = row[row <= -min_cycles].sort_values().head(top)
            items = []
            for sensor, value in neg.items():
                short, desc, comp = SENSOR_INFO.get(sensor, ("age", "Cycles in service", "Whole engine"))
                items.append({"sensor": sensor, "name": short, "description": desc, "component": comp,
                              "shap_cycles": round(float(value), 1)})
            out.append(items)
        return out


def get_explainer(bundle: dict) -> RULExplainer | None:
    """One explainer per loaded bundle (cached on the bundle; never saved to disk)."""
    if "_explainer" not in bundle:
        try:
            bundle["_explainer"] = RULExplainer(bundle["model"])
        except Exception as exc:  # SHAP missing or unsupported model: explanations are optional
            log.warning("SHAP explanations disabled: %s", exc)
            bundle["_explainer"] = None
    return bundle["_explainer"]
