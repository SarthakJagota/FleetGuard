import numpy as np
import pytest

from fleetguard.agent.triage import TriageAgent, TriageThresholds
from fleetguard.models.risk import FailureRisk, calibration_table, health_band, health_score, risk_metrics


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0, 125, n)
    pred = np.clip(true + rng.normal(0, 8, n), 0, 125)
    anom = np.clip(3 * np.exp(-true / 25) + rng.normal(0, 0.2, n), 0, None)
    return pred, anom, true


def test_health_score_formula():
    assert health_score([125], [0.5])[0] == 100        # full RUL, no anomaly
    assert health_score([0], [3.0])[0] == 0            # failed, strongly anomalous
    assert health_score([62.5], [1.0])[0] == pytest.approx(50)    # half the life left, no anomaly
    assert health_score([125], [3.0])[0] == 75         # anomaly alone costs up to 25%
    assert health_score([5], [0.5])[0] < 5             # near failure is never 'Fair' or 'Poor'
    assert [health_band(s) for s in (95, 60, 30, 10)] == ["Good", "Fair", "Poor", "Critical"]


def test_failure_probability_is_monotone_and_calibrated():
    pred, anom, true = synthetic()
    risk = FailureRisk(horizon=30).fit(pred[:3000], anom[:3000], true[:3000])
    p = risk.predict_proba(pred[3000:], anom[3000:])
    assert ((p >= 0) & (p <= 1)).all()
    # Lower predicted RUL -> higher failure probability.
    grid = risk.predict_proba(np.array([5, 20, 40, 80, 120]), np.zeros(5))
    assert (np.diff(grid) < 0).all()
    m = risk_metrics(p, risk.health(pred[3000:], anom[3000:]), true[3000:], 30, 125)
    assert m["failure_prob_auc"] > 0.95
    assert m["failure_prob_ece"] < 0.05           # predicted % ~ observed %
    assert m["failure_prob_brier"] < 0.08
    assert m["health_spearman_rul"] > 0.9
    cal = calibration_table(p, true[3000:] <= 30)
    assert cal["n"].sum() == 3000


def test_failure_risk_needs_both_classes():
    with pytest.raises(ValueError):
        FailureRisk().fit([100, 110], [0, 0], [100, 120])


def test_agent_escalates_on_high_failure_probability():
    agent = TriageAgent(TriageThresholds(critical_rul=15, high_rul=30, medium_rul=60, high_failure_prob=0.5))
    base = dict(engine_id=1, cycle=150, n_cycles=150, rul_pred=45, rul_lower=32, rul_upper=60,
                anomaly_index=0.8, anomaly_alarm=False, anomaly_run=0, imputed_fraction=0.0, health_score=55.0)
    low_p = agent.decide({**base, "failure_prob": 0.2})
    high_p = agent.decide({**base, "failure_prob": 0.7})
    assert low_p.severity == "MEDIUM"
    assert high_p.severity == "HIGH" and "FAILURE_PROB_HIGH" in high_p.reason_codes
    assert high_p.failure_prob == 0.7 and high_p.health_score == 55.0
    # Missing probability (older model bundle) simply skips the rule.
    assert agent.decide({**base, "failure_prob": float("nan")}).severity == "MEDIUM"
