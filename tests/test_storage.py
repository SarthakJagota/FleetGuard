"""SQLite storage: work-order round trip, history, migration, scoring history."""

import json

import pandas as pd

from fleetguard.agent.triage import TriageAgent, TriageThresholds
from fleetguard.agent.work_orders import WorkOrderStore
from fleetguard.storage import Database

AGENT = TriageAgent(TriageThresholds())


def decision(engine=7, cycle=150, rul=50.0, lo=35.0, hi=65.0):
    return AGENT.decide(dict(engine_id=engine, cycle=cycle, n_cycles=cycle, rul_pred=rul, rul_lower=lo, rul_upper=hi,
                             anomaly_index=0.5, anomaly_alarm=False, failure_prob=0.1, health_score=40.0))


def test_orders_round_trip_with_history(tmp_path):
    db_path = tmp_path / "fg.db"
    store = WorkOrderStore(db_path)
    _, wo = store.apply("FD001", decision(), "v1")
    store.apply("FD001", decision(cycle=160, rul=10, lo=3, hi=18), "v1")       # escalates
    store.set_status(wo["wo_id"], "ACKNOWLEDGED", note="parts ordered", by="tech-1")
    store.save()

    again = WorkOrderStore(db_path)
    assert len(again.orders) == 1
    got = again.orders[0]
    assert got["severity"] == "CRITICAL" and got["status"] == "ACKNOWLEDGED" and got["health_score"] == 40.0
    assert [e["event"] for e in got["history"]] == ["CREATED", "ESCALATED", "STATUS_ACKNOWLEDGED"]
    assert got["history"][-1]["by"] == "tech-1" and got["history"][-1]["note"] == "parts ordered"
    # Key fields are real columns, so they can be queried with SQL.
    row = Database(db_path).query("SELECT fleet, engine_id, severity, status FROM work_orders").iloc[0].to_dict()
    assert row == {"fleet": "FD001", "engine_id": 7, "severity": "CRITICAL", "status": "ACKNOWLEDGED"}


def test_only_changed_orders_are_written(tmp_path):
    store = WorkOrderStore(tmp_path / "fg.db")
    for e in range(1, 4):
        store.apply("FD001", decision(engine=e), "v1")
    store.save()
    assert store._dirty == set()
    store.set_status("WO-00002", "CLOSED")
    assert store._dirty == {"WO-00002"}
    store.save()
    statuses = Database(tmp_path / "fg.db").query("SELECT wo_id, status FROM work_orders ORDER BY wo_id")
    assert statuses["status"].tolist() == ["OPEN", "CLOSED", "OPEN"]


def test_two_writers_do_not_lose_updates(tmp_path):
    """Dashboard and pipeline may hold separate stores; each writes only its own changes."""
    path = tmp_path / "fg.db"
    seed = WorkOrderStore(path)
    seed.apply("FD001", decision(engine=1), "v1")
    seed.apply("FD001", decision(engine=2), "v1")
    seed.save()
    dashboard, pipeline = WorkOrderStore(path), WorkOrderStore(path)
    dashboard.set_status("WO-00001", "CLOSED", note="done")
    dashboard.save()
    pipeline.apply("FD001", decision(engine=2, cycle=170, rul=8, lo=2, hi=15), "v1")
    pipeline.save()
    final = {o["wo_id"]: o for o in WorkOrderStore(path).orders}
    assert final["WO-00001"]["status"] == "CLOSED"          # not overwritten by the pipeline
    assert final["WO-00002"]["severity"] == "CRITICAL"


def test_migrates_legacy_json_once(tmp_path):
    legacy = tmp_path / "work_orders.json"
    js = WorkOrderStore(legacy)
    js.apply("FD003", decision(engine=4), "v1")
    js.save()
    store = WorkOrderStore(tmp_path / "fg.db", import_json=legacy)
    assert len(store.orders) == 1 and store.orders[0]["subset"] == "FD003"
    assert not legacy.exists() and (tmp_path / "work_orders.json.migrated").exists()
    assert len(WorkOrderStore(tmp_path / "fg.db").orders) == 1


def test_remove_fleets(tmp_path):
    store = WorkOrderStore(tmp_path / "fg.db")
    store.apply("FD001", decision(engine=1), "v1")
    store.apply("UPLOAD-x", decision(engine=1), "v1")
    store.save()
    store.remove_fleets(["FD001"])
    assert [o["subset"] for o in WorkOrderStore(tmp_path / "fg.db").orders] == ["UPLOAD-x"]


def test_scoring_history(tmp_path):
    db = Database(tmp_path / "fg.db")
    fleet = pd.DataFrame({"engine_id": [1, 2], "cycle": [100, 120], "severity": ["LOW", "HIGH"],
                          "rul_pred": [90.0, 25.0], "rul_lower": [80.0, 18.0], "rul_upper": [100.0, 33.0],
                          "failure_prob": [0.01, float("nan")], "health_score": [70.0, 18.0],
                          "anomaly_index": [0.4, 1.3], "true_rul": [95, 22], "wo_id": [None, "WO-00001"]})
    r1 = db.record_scoring("FD001", "v1", fleet)
    r2 = db.record_scoring("FD001", "v2", fleet.assign(cycle=fleet["cycle"] + 5))
    runs = db.scoring_runs("FD001")
    assert runs["run_id"].tolist() == [r2, r1]
    assert json.loads(runs.iloc[0]["severity_counts"]) == {"LOW": 1, "HIGH": 1}
    hist = db.engine_history("FD001", 2)
    assert hist["cycle"].tolist() == [120, 125] and hist["model_version"].tolist() == ["v1", "v2"]
    assert pd.isna(hist["failure_prob"]).all()            # NaN stored as NULL
    assert db.counts()["predictions"] == 4
