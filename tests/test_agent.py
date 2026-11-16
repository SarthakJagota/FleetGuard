import pandas as pd
import pytest

from fleetguard.agent.triage import TriageAgent, TriageThresholds, calibrate_high_threshold
from fleetguard.agent.work_orders import WorkOrderStore, render_markdown

AGENT = TriageAgent(TriageThresholds(critical_rul=15, high_rul=30, medium_rul=60))


def state(**kw):
    base = dict(engine_id=7, cycle=150, n_cycles=150, rul_pred=100, rul_lower=85, rul_upper=120,
                anomaly_index=0.5, anomaly_alarm=False, anomaly_run=0, imputed_fraction=0.0)
    base.update(kw)
    return base


@pytest.mark.parametrize("kw,severity,code", [
    (dict(rul_pred=10, rul_lower=2, rul_upper=20), "CRITICAL", "RUL_CRITICAL"),
    (dict(rul_pred=25, rul_lower=18, rul_upper=35), "HIGH", "RUL_HIGH"),
    (dict(rul_pred=40, rul_lower=12, rul_upper=60), "HIGH", "RUL_LOWER_BOUND"),
    (dict(rul_pred=55, rul_lower=40, rul_upper=70, anomaly_alarm=True, anomaly_run=5), "HIGH", "ANOMALY_AND_RUL"),
    (dict(rul_pred=50, rul_lower=35, rul_upper=65), "MEDIUM", "RUL_MEDIUM"),
    (dict(rul_pred=80, rul_lower=65, rul_upper=95, anomaly_alarm=True, anomaly_run=4), "MEDIUM", "ANOMALY_ALARM"),
    (dict(rul_pred=118, rul_lower=100, rul_upper=125, anomaly_alarm=True), "REVIEW", "MODEL_DISAGREEMENT"),
    (dict(rul_pred=90, rul_lower=20, rul_upper=125), "REVIEW", "HIGH_UNCERTAINTY"),
    (dict(imputed_fraction=0.5), "REVIEW", "DATA_QUALITY"),
    (dict(rul_pred=float("nan")), "REVIEW", "INVALID_PREDICTION"),
    (dict(n_cycles=8, rul_pred=20, rul_lower=10), "REVIEW", "INSUFFICIENT_HISTORY"),
    (dict(n_cycles=8), "LOW", "INSUFFICIENT_HISTORY"),
    (dict(), "LOW", "HEALTHY"),
])
def test_triage_rules(kw, severity, code):
    d = AGENT.decide(state(**kw))
    assert d.severity == severity
    assert code in d.reason_codes


def test_due_window_and_indicators():
    dev = pd.Series({"sensor_11": 3.2, "sensor_4": 2.5, "sensor_12": -2.9, "sensor_7": 0.4})
    d = AGENT.decide(state(rul_pred=25, rul_lower=18, rul_upper=35), dev)
    assert d.due_within_cycles == 13  # lower bound 18 - 5 cycles safety margin
    assert [i["name"] for i in d.indicators] == ["Ps30", "phi", "T50"]
    assert any("HPC" in c for c in d.checks)
    assert AGENT.decide(state(rul_pred=5, rul_lower=0, rul_upper=12)).due_within_cycles == 0


def test_calibration_hits_target_recall():
    true = list(range(0, 120))
    pred = [t + 8 for t in true]  # model over-estimates by 8 cycles
    c = calibrate_high_threshold(pred, true, degraded=30, target_recall=0.95, bounds=(25, 50))
    assert c["val_recall"] >= 0.95 and 25 <= c["high_rul"] <= 50
    assert c["high_rul"] >= 30 + 8 - 2  # compensates for the optimistic bias


def test_work_order_lifecycle(tmp_path):
    store = WorkOrderStore(tmp_path / "wo.json")
    low = AGENT.decide(state())
    assert store.apply("FD001", low, "v1") == ("NO_ACTION", None)

    med = AGENT.decide(state(rul_pred=50, rul_lower=35, rul_upper=65))
    event, wo = store.apply("FD001", med, "v1")
    assert event == "CREATED" and wo["type"] == "INSPECTION" and wo["priority"] == "P3"

    # Same engine again at the next cycle -> update, no duplicate.
    event, _ = store.apply("FD001", AGENT.decide(state(cycle=151, rul_pred=49, rul_lower=34, rul_upper=64)), "v1")
    assert event == "UPDATED" and len(store.orders) == 1

    # Degrades -> escalated in place.
    event, wo = store.apply("FD001", AGENT.decide(state(cycle=170, rul_pred=12, rul_lower=4, rul_upper=20)), "v1")
    assert event == "ESCALATED" and wo["severity"] == "CRITICAL" and wo["priority"] == "P1" and len(store.orders) == 1

    # Re-scoring the same cycle (e.g. with a newer model) refreshes evidence without a history entry.
    n_events = len(wo["history"])
    store.apply("FD001", AGENT.decide(state(cycle=170, rul_pred=11, rul_lower=3, rul_upper=19)), "v2")
    assert wo["model_version"] == "v2" and wo["rul_pred"] == 11 and len(wo["history"]) == n_events

    # A better reading never auto-downgrades an open order.
    store.apply("FD001", AGENT.decide(state(cycle=171, rul_pred=50, rul_lower=35, rul_upper=65)), "v1")
    assert store.orders[0]["severity"] == "CRITICAL"

    # Human closes it; persistence round-trips; a new problem opens a new order.
    store.set_status(wo["wo_id"], "CLOSED", note="compressor replaced")
    store.save()
    reloaded = WorkOrderStore(tmp_path / "wo.json")
    assert reloaded.orders[0]["status"] == "CLOSED"
    event, wo2 = reloaded.apply("FD001", med, "v2")
    assert event == "CREATED" and wo2["wo_id"] == "WO-00002"
    assert "Why this order was raised" in render_markdown(wo2)


def test_separate_engines_get_separate_orders(tmp_path):
    store = WorkOrderStore(tmp_path / "wo.json")
    for e in (1, 2, 3):
        store.apply("FD002", AGENT.decide(state(engine_id=e, rul_pred=10, rul_lower=3, rul_upper=20)), "v1")
    assert len(store.orders) == 3
    with pytest.raises(ValueError):
        store.set_status("WO-00001", "DONE")
