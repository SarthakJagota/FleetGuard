"""SHAP explanations: correctness (additivity) and their use in work orders."""

import numpy as np
import pandas as pd
import pytest

from fleetguard.agent.work_orders import WorkOrderStore, render_markdown
from fleetguard.data.loader import add_train_rul
from fleetguard.explain import RULExplainer, feature_sensor, get_explainer
from fleetguard.models.rul import RULModel, make_model
from fleetguard.scoring import score_fleet


@pytest.fixture(scope="module")
def last_rows(fast_bundle, fd001):
    feats = fast_bundle["features"].transform(fd001["test"][fd001["test"]["engine_id"] <= 15])
    return feats.loc[feats.groupby("engine_id")["cycle"].idxmax()]


def test_feature_to_sensor_mapping():
    assert feature_sensor("sensor_11_slope30") == "sensor_11"
    assert feature_sensor("sensor_4_ewm") == "sensor_4"
    assert feature_sensor("cycle") == "cycle"


def test_tree_shap_is_additive(fast_bundle, last_rows):
    """base value + sum of contributions == the model's raw output, for every row."""
    ex = RULExplainer(fast_bundle["model"])
    contrib = ex.contributions(last_rows)
    raw = fast_bundle["model"].estimator.predict(last_rows[ex.features])
    np.testing.assert_allclose(ex.base_value + contrib.sum(axis=1), raw, atol=1e-3)
    per_sensor = ex.by_sensor(contrib)
    np.testing.assert_allclose(per_sensor.sum(axis=1), contrib.sum(axis=1), atol=1e-6)
    assert "cycle" in per_sensor.columns and "sensor_11" in per_sensor.columns


def test_linear_explanation_is_additive(fast_bundle, fd001):
    fp = fast_bundle["features"]
    tr = fp.transform(add_train_rul(fd001["train"][fd001["train"]["engine_id"] <= 20], cap=125))
    est = make_model("ridge").fit(tr[fp.feature_names_], tr["rul_capped"])
    ex = RULExplainer(RULModel("ridge", est, fp.feature_names_, 125))
    X = tr.tail(5)
    np.testing.assert_allclose(ex.base_value + ex.contributions(X).sum(axis=1), est.predict(X[fp.feature_names_]), atol=1e-6)


def test_drivers_are_the_most_negative_sensors(fast_bundle, last_rows):
    ex = get_explainer(fast_bundle)
    assert get_explainer(fast_bundle) is ex  # cached on the bundle
    per_sensor = ex.by_sensor(ex.contributions(last_rows))
    for (_, row), items in zip(per_sensor.iterrows(), ex.drivers(last_rows, top=3)):
        assert len(items) <= 3 and all(x["shap_cycles"] <= -0.5 for x in items)
        expected = row[row <= -0.5].sort_values().head(3).index.tolist()
        assert [x["sensor"] for x in items] == expected


def test_work_orders_carry_shap_drivers(fast_bundle, fd001, tmp_path):
    store = WorkOrderStore(tmp_path / "wo.json")
    fleet, _, _ = score_fleet(fast_bundle, fd001["test"], store)
    ordered = fleet[fleet["severity"] != "LOW"]
    assert len(ordered) > 0
    assert (ordered["model_drivers"].str.len() > 0).mean() > 0.9   # nearly all orders are explained
    assert (fleet.loc[fleet["severity"] == "LOW", "model_drivers"] == "").all()  # not computed when no order
    wo = next(o for o in store.orders if o["model_drivers"])
    md = render_markdown(wo)
    assert "What drove the RUL prediction down (SHAP)" in md and "not a confirmed root cause" in md
