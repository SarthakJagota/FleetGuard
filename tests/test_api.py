"""REST API: runs against a small model trained in a temp directory (never the real outputs)."""

import io

import pytest
from fastapi.testclient import TestClient

from fleetguard import api
from fleetguard.pipeline import FleetGuard


@pytest.fixture(scope="module")
def client(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("api")
    c = cfg.override(paths={"raw_data": str(cfg.path("raw_data")), "incoming_data": str(tmp / "in"),
                            "models": str(tmp / "models"), "outputs": str(tmp / "out"),
                            "database": str(tmp / "out" / "fleetguard.db")})
    fg = FleetGuard(c, track=False, fast=True, candidates=["lightgbm"])
    fg.train("FD001")
    fg.score("FD001")
    old_cfg = api.CFG
    api.CFG, api._bundles = c, {}
    yield TestClient(api.app)
    api.CFG, api._bundles = old_cfg, {}


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models_available"] == ["FD001"]
    m = client.get("/models").json()
    assert m[0]["subset"] == "FD001" and "test_rmse" in m[0]["test_metrics"]


def test_example_then_predict_round_trip(client):
    body = client.get("/example/FD001", params={"engine_id": 31, "last_cycles": 500}).json()
    assert len(body["readings"]) > 15
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"]["subset"] == "FD001" and "single operating regime" in out["model"]["selection_reason"]
    assert out["model"]["regime_mismatch"] is False
    e = out["engines"][0]
    assert e["engine_id"] == 31 and e["severity"] in {"CRITICAL", "HIGH", "MEDIUM", "REVIEW", "LOW"}
    assert e["rul_lower"] <= e["rul_pred"] <= e["rul_upper"]
    assert 0 <= e["failure_prob"] <= 1 and 0 <= e["health_score"] <= 100
    assert e["work_order"] is None                        # not requested


def test_predict_creates_and_updates_work_orders(client):
    body = client.get("/example/FD001", params={"engine_id": 34, "last_cycles": 500}).json()  # near failure
    body.update(create_work_orders=True, fleet="API-TEST")
    first = client.post("/predict", json=body).json()["engines"][0]
    assert first["severity"] in {"CRITICAL", "HIGH"} and first["model_drivers"]   # SHAP explained
    assert first["work_order"]["event"] == "CREATED"
    again = client.post("/predict", json=body).json()["engines"][0]
    assert again["work_order"] == {"wo_id": first["work_order"]["wo_id"], "event": "UPDATED"}   # no duplicate

    wo_id = first["work_order"]["wo_id"]
    listed = client.get("/work-orders", params={"fleet": "API-TEST"}).json()
    assert [o["wo_id"] for o in listed] == [wo_id]
    detail = client.get(f"/work-orders/{wo_id}", params={"markdown": True}).json()
    assert "What drove the RUL prediction down (SHAP)" in detail["markdown"]
    upd = client.patch(f"/work-orders/{wo_id}", json={"status": "CLOSED", "note": "fixed", "by": "tech-7"}).json()
    assert upd["status"] == "CLOSED" and upd["history"][-1]["by"] == "tech-7"
    assert client.get("/work-orders", params={"fleet": "API-TEST", "status": "OPEN"}).json() == []


def test_predict_csv_upload(client, fd001):
    eng = fd001["test"][fd001["test"]["engine_id"].isin([1, 2])]
    csv = eng.to_csv(index=False).encode()
    r = client.post("/predict/csv", files={"file": ("two_engines.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 200, r.text
    assert [e["engine_id"] for e in r.json()["engines"]] == [1, 2]


def test_input_errors(client):
    bad = {"readings": [{"engine_id": 1, "cycle": 1, "sensor_4": 1400.0}]}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422 and "missing columns" in r.json()["detail"]
    assert client.post("/predict", json={"readings": []}).status_code == 422
    body = client.get("/example/FD001").json()
    assert client.post("/predict", json={**body, "model": "FD004"}).status_code == 404   # not trained here
    assert client.get("/work-orders/WO-99999").status_code == 404
    assert client.get("/example/FD009").status_code == 404


def test_fleet_endpoint(client):
    f = client.get("/fleets/FD001").json()
    assert f["engines"] == 100 and sum(f["severity_counts"].values()) == 100
    crit = client.get("/fleets/FD001", params={"severity": "critical"}).json()
    assert all(e["severity"] == "CRITICAL" for e in crit["fleet"])
