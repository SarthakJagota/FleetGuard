import numpy as np
import pandas as pd
import pytest

from fleetguard.mlops.drift import psi
from fleetguard.models.anomaly import METHODS, AnomalyDetector
from fleetguard.models.metrics import nasa_score, regression_metrics, warning_metrics
from fleetguard.models.rul import ConformalIntervals, imbalance_weights


def test_nasa_score_penalises_late_predictions_more():
    early = nasa_score([50], [40])   # predicted 10 cycles too early
    late = nasa_score([50], [60])    # predicted 10 cycles too late
    assert late > early > 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_regression_metrics_perfect():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1


def test_imbalance_weights_upweight_rare_low_rul():
    y = np.array([125] * 900 + [10] * 100, float)
    w = imbalance_weights(y, [0, 25, 50, 75, 100, 126], power=0.5)
    assert w.mean() == pytest.approx(1.0)
    assert w[-1] > w[0]
    assert np.allclose(imbalance_weights(y, [0, 126], power=0), 1)


def test_conformal_intervals_reach_target_coverage():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 125, 4000)
    pred = true + rng.normal(0, 5 + 0.1 * true)  # error grows with RUL
    ci = ConformalIntervals(coverage=0.8).fit(pred[:2000], true[:2000])
    lo, hi = ci.predict(pred[2000:])
    cover = np.mean((true[2000:] >= lo) & (true[2000:] <= hi))
    assert 0.75 < cover < 0.85
    # Mondrian bands: intervals are narrower near failure than far from it.
    w_near = np.mean((hi - lo)[pred[2000:] < 30])
    w_far = np.mean((hi - lo)[pred[2000:] > 90])
    assert w_near < w_far


def test_warning_metrics_imbalanced():
    rul = np.array([5, 10, 50, 100, 120, 125, 125, 125])
    score = np.array([0.9, 0.8, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1])
    m = warning_metrics(rul, score, score > 0.5, horizon=30)
    assert m["recall"] == 1 and m["precision"] == 1 and m["pr_auc"] == pytest.approx(1)


@pytest.fixture(scope="module")
def detector():
    """Healthy data with correlated sensors (like real engines): b follows a, c is independent."""
    rng = np.random.default_rng(1)
    cols = ["a_ewm", "b_ewm", "c_ewm"]

    def healthy(n, eid):
        a = rng.normal(0, 1, n)
        return pd.DataFrame({"a_ewm": a, "b_ewm": a + rng.normal(0, 0.2, n), "c_ewm": rng.normal(0, 1, n)}).assign(engine_id=eid)

    det = AnomalyDetector(cols, threshold_quantile=0.99, persistence=3).fit(healthy(3000, 1), healthy(800, 2))
    return det, healthy(800, 4), rng


@pytest.mark.parametrize("method", METHODS)
def test_anomaly_detectors_flag_drifting_engine(detector, method):
    det, healthy_val, rng = detector
    n = 60
    drift = np.linspace(0, 6, n)[:, None]
    drifting = pd.DataFrame(rng.normal(0, 0.3, (n, 3)) + drift, columns=det.columns).assign(engine_id=3)
    s = det.score(drifting, method)
    assert s["anomaly_alarm"].iloc[-10:].all()
    assert not s["anomaly_alarm"].iloc[:10].any()
    assert det.score(healthy_val, method)["anomaly_alarm"].mean() < 0.02


@pytest.mark.parametrize("method", ["mahalanobis", "autoencoder"])
def test_detectors_catch_broken_sensor_relationship(detector, method):
    """Each sensor stays in its normal range, but b no longer follows a (e.g. a failing sensor or seal).
    Detectors that learn the joint healthy pattern catch it; a per-sensor range check would not."""
    det, _, rng = detector
    a = rng.uniform(-1.5, 1.5, 40)
    broken = pd.DataFrame({"a_ewm": a, "b_ewm": -a, "c_ewm": rng.normal(0, 1, 40)}).assign(engine_id=5)
    assert broken[["a_ewm", "b_ewm"]].abs().max().max() <= 1.5          # nothing out of range on its own
    s = det.score(broken[np.abs(broken["a_ewm"]) > 0.8], method)          # where the mismatch is clear
    assert (s["anomaly_index"] > 1).mean() > 0.9


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1, 1, 5000)) > 0.2
