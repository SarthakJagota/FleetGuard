import numpy as np
import pandas as pd
import pytest

from fleetguard.mlops.drift import psi
from fleetguard.models.anomaly import AnomalyDetector
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


def test_anomaly_detector_flags_shifted_engine():
    rng = np.random.default_rng(1)
    cols = ["a_ewm", "b_ewm", "c_ewm"]
    healthy = pd.DataFrame(rng.normal(0, 1, (2000, 3)), columns=cols).assign(engine_id=1)
    healthy_val = pd.DataFrame(rng.normal(0, 1, (500, 3)), columns=cols).assign(engine_id=2)
    det = AnomalyDetector(cols, threshold_quantile=0.99, persistence=3).fit(healthy, healthy_val)
    drifting = pd.DataFrame(rng.normal(0, 1, (60, 3)) + np.linspace(0, 6, 60)[:, None], columns=cols).assign(engine_id=3)
    s = det.score(drifting)
    assert s["anomaly_alarm"].iloc[-10:].all()
    assert not s["anomaly_alarm"].iloc[:10].any()
    assert det.score(healthy_val)["anomaly_alarm"].mean() < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1, 1, 5000)) > 0.2
