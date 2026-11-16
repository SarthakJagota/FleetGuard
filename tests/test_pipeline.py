"""End-to-end integration test on FD001 with small, fast models and no MLflow."""

import json

import pandas as pd

from fleetguard.agent.work_orders import WorkOrderStore
from fleetguard.pipeline import FleetGuard


def test_closed_loop_end_to_end(tmp_cfg):
    fg = FleetGuard(tmp_cfg, track=False, fast=True, candidates=["ridge", "lightgbm"])

    v1 = fg.train("FD001", initial_fraction=0.7)
    assert v1["version"] == "v1" and v1["decision"].startswith("promoted")
    assert v1["metrics"]["test_rmse"] < 35

    # No new data -> retraining is skipped.
    assert fg.retrain("FD001")["decision"].startswith("skipped")

    batch = fg.simulate_arrival("FD001")
    assert batch.exists()
    v2 = fg.retrain("FD001")
    assert v2["version"] == "v2"
    assert "drifted_features" in v2
    assert not batch.exists()  # moved to the archive once ingested

    reg = fg.registry("FD001")
    assert [v["version"] for v in reg.versions()] == ["v1", "v2"]
    assert reg.load()["features"].fitted_

    fleet = fg.score("FD001")
    assert len(fleet) == 100
    assert set(fleet["severity"]) <= {"CRITICAL", "HIGH", "MEDIUM", "LOW", "REVIEW"}
    out = tmp_cfg.path("outputs") / "FD001"
    for name in ("fleet_latest.csv", "fleet_history.csv.gz", "monitoring.json", "drift_fleet.csv",
                 "evaluation/metrics.json", "evaluation/test_predictions.csv"):
        assert (out / name).exists(), name

    store = WorkOrderStore(tmp_cfg.path("outputs") / "work_orders.json")
    n_orders = len(store.orders)
    assert n_orders == (fleet["severity"] != "LOW").sum()
    assert len(list((out / "work_orders").glob("WO-*.md"))) == n_orders

    # Re-scoring the same fleet is idempotent: orders are updated, not duplicated.
    fg.score("FD001")
    assert len(WorkOrderStore(tmp_cfg.path("outputs") / "work_orders.json").orders) == n_orders

    # Streaming replay produces work orders over time with escalations.
    events = fg.stream("FD001", step=25)
    assert len(events) > 0 and "CREATED" in set(events["event"])
    monitoring = json.loads((out / "monitoring.json").read_text())
    assert monitoring["engines"] == 100
