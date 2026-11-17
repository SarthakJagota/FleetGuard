"""FleetGuard REST API (FastAPI).

Run:  fleetguard serve            (or: uvicorn fleetguard.api:app)
Docs: http://127.0.0.1:8000/docs  (interactive, generated from this file)

Endpoints
    GET   /health                      liveness + which models are loaded
    GET   /models                      champion model per subset with test metrics
    GET   /example/{subset}            a ready-to-send /predict body built from real data
    POST  /predict                     sensor history (JSON) -> RUL, risk, triage decision
    POST  /predict/csv                 same, for an uploaded CSV or NASA .txt file
    GET   /fleets/{subset}             latest scored state of a benchmark fleet
    GET   /work-orders                 list work orders (filters: fleet, status, severity)
    GET   /work-orders/{wo_id}         one work order with its history
    PATCH /work-orders/{wo_id}         acknowledge / close / reject (the human in the loop)

The API reuses exactly the same code as the pipeline and dashboard (features,
models, agent, work-order store), so all three always agree.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from . import __version__, inference
from .agent.work_orders import WorkOrderStore, render_markdown
from .config import SUBSETS, load_config
from .data.loader import RAW_COLS, load_subset
from .mlops.registry import ModelRegistry
from .models.risk import health_band
from .scoring import score_history, triage_rows

CFG = load_config(os.environ.get("FLEETGUARD_CONFIG"))
_bundles: dict[tuple[str, str], dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    readings: list[dict[str, float]] = Field(
        ..., min_length=1,
        description="Cycle-by-cycle rows for one or more engines. Each row needs engine_id, cycle, setting_1-3 and "
                    "sensor_1-21. Send each engine's whole history: the model uses 10-30 cycle trends.",
    )
    model: str = Field("auto", description="'auto' picks the model from the data (operating regimes); or FD001-FD004")
    create_work_orders: bool = Field(False, description="Also create/update work orders in the database")
    fleet: str = Field("API", description="Fleet name used for work orders (keeps engine ids from colliding)")

    model_config = {"json_schema_extra": {"example": {
        "model": "auto", "create_work_orders": False, "fleet": "API",
        "readings": [{"engine_id": 1, "cycle": 1, "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
                      **{f"sensor_{i}": v for i, v in enumerate(
                          [518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66,
                           2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.419], start=1)}}],
    }}}


class ModelInfo(BaseModel):
    subset: str
    version: str
    algorithm: str
    selection_reason: str
    regime_mismatch: bool


class Driver(BaseModel):
    sensor: str
    name: str
    description: str
    component: str
    shap_cycles: float


class Indicator(BaseModel):
    sensor: str
    name: str
    description: str
    component: str
    deviation_sd: float
    direction: str


class WorkOrderRef(BaseModel):
    wo_id: str
    event: Literal["CREATED", "ESCALATED", "UPDATED"]


class EnginePrediction(BaseModel):
    engine_id: int
    cycle: int
    cycles_observed: int
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "REVIEW", "LOW"]
    action: str
    due_within_cycles: int | None
    rul_pred: float
    rul_lower: float
    rul_upper: float
    health_score: float | None
    health_band: str | None
    failure_prob: float | None = Field(None, description="P(failure within 30 cycles)")
    anomaly_index: float
    anomaly_alarm: bool
    reason_codes: list[str]
    reasons: list[str]
    indicators: list[Indicator] = Field(description="Sensors deviating most from the healthy baseline")
    model_drivers: list[Driver] = Field(description="SHAP: sensors that lowered the predicted RUL (orders only)")
    work_order: WorkOrderRef | None = None


class PredictResponse(BaseModel):
    model: ModelInfo
    data_quality: dict
    engines: list[EnginePrediction]


class StatusUpdate(BaseModel):
    status: Literal["ACKNOWLEDGED", "CLOSED", "REJECTED", "OPEN"]
    note: str = ""
    by: str = "api"


# ---------------------------------------------------------------- helpers
def _available() -> list[str]:
    return [s for s in SUBSETS if ModelRegistry(CFG.path("models"), s).champion]


def _bundle(subset: str) -> tuple[dict, str]:
    """Champion bundle for a subset, cached; reloaded automatically after a promotion."""
    version = ModelRegistry(CFG.path("models"), subset).champion
    if version is None:
        raise HTTPException(503, f"No trained model for {subset}. Run `fleetguard train --subset {subset}`.")
    key = (subset, version)
    with _lock:
        if key not in _bundles:
            _bundles[key] = ModelRegistry(CFG.path("models"), subset).load(version)
        return _bundles[key], version


def _store() -> WorkOrderStore:
    return WorkOrderStore(CFG.path("database"), import_json=CFG.path("outputs") / "work_orders.json")


def _clean(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def _predict(raw: pd.DataFrame, choice: str, create_work_orders: bool, fleet: str) -> PredictResponse:
    available = _available()
    if not available:
        raise HTTPException(503, "No trained models. Run `fleetguard run-all` first.")
    if choice.lower() == "auto":
        subset, reason = inference.suggest_subset(raw, available)
    else:
        subset, reason = choice.upper(), "selected by the caller"
        if subset not in available:
            raise HTTPException(404, f"Model {subset} is not trained; available: {available}")
    bundle, version = _bundle(subset)
    mismatch = inference.is_multi_regime(raw) != (bundle["features"].n_conditions > 1)

    try:
        hist, feats, quality = score_history(bundle, raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    latest = hist.groupby("engine_id")["cycle"].idxmax().sort_values().tolist()
    decisions = triage_rows(bundle, hist, feats, latest)

    refs: dict[int, WorkOrderRef] = {}
    if create_work_orders:
        store = _store()
        for d in decisions:
            event, wo = store.apply(fleet, d, f"{subset}-{version}")
            if wo is not None:
                refs[d.engine_id] = WorkOrderRef(wo_id=wo["wo_id"], event=event)
        store.save()

    engines = []
    for d, i in zip(decisions, latest):
        engines.append(EnginePrediction(
            engine_id=d.engine_id, cycle=d.cycle, cycles_observed=int(hist.at[i, "n_cycles"]),
            severity=d.severity, action=d.action, due_within_cycles=d.due_within_cycles,
            rul_pred=d.rul_pred, rul_lower=d.rul_lower, rul_upper=d.rul_upper,
            health_score=_clean(d.health_score),
            health_band=health_band(d.health_score) if _clean(d.health_score) is not None else None,
            failure_prob=_clean(d.failure_prob), anomaly_index=d.anomaly_index, anomaly_alarm=d.anomaly_alarm,
            reason_codes=d.reason_codes, reasons=d.reasons,
            indicators=[Indicator(**x) for x in d.indicators], model_drivers=[Driver(**x) for x in d.drivers],
            work_order=refs.get(d.engine_id),
        ))
    model = ModelInfo(subset=subset, version=version, algorithm=bundle["model"].name, selection_reason=reason,
                      regime_mismatch=mismatch)
    return PredictResponse(model=model, data_quality=quality, engines=engines)


# ---------------------------------------------------------------- app
app = FastAPI(
    title="FleetGuard API",
    version=__version__,
    description="Predictive maintenance for a turbofan fleet: remaining-useful-life prediction with calibrated "
                "intervals, failure probability, health score, anomaly detection and a rule-based triage agent that "
                "raises maintenance work orders. Start with **GET /example/FD001**, then paste its body into "
                "**POST /predict**.",
)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "version": __version__, "models_available": _available(),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/models", tags=["system"])
def models() -> list[dict]:
    out = []
    for s in _available():
        reg = ModelRegistry(CFG.path("models"), s)
        v = next(x for x in reg.versions() if x["version"] == reg.champion)
        m = v.get("metrics", {})
        out.append({"subset": s, "version": reg.champion, "algorithm": v.get("model_name"),
                    "trained_engines": v.get("trained_engines"),
                    "test_metrics": {k: round(m[k], 4) for k in ("test_rmse", "test_mae", "test_r2", "test_nasa_score",
                                                                 "test_interval_coverage", "test_failure_prob_auc")
                                     if k in m}})
    return out


@app.get("/example/{subset}", tags=["predict"], response_model=PredictRequest)
def example(subset: str, engine_id: int = Query(1, ge=1), last_cycles: int = Query(60, ge=1, le=600)) -> dict:
    """A real engine's history from the NASA test set, formatted as a /predict request body."""
    subset = subset.upper()
    if subset not in SUBSETS:
        raise HTTPException(404, f"Unknown subset {subset}")
    test = load_subset(CFG.path("raw_data"), subset)["test"]
    eng = test[test["engine_id"] == engine_id]
    if eng.empty:
        raise HTTPException(404, f"Engine {engine_id} not in the {subset} test set")
    rows = eng.tail(last_cycles)[RAW_COLS].astype(float).to_dict("records")
    return {"model": "auto", "create_work_orders": False, "fleet": "API", "readings": rows}


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest) -> PredictResponse:
    """Predict RUL, risk and the triage decision for the latest cycle of every engine in the request."""
    raw = pd.DataFrame(req.readings)
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise HTTPException(422, f"Readings are missing columns: {missing}")
    return _predict(raw[RAW_COLS], req.model, req.create_work_orders, req.fleet)


@app.post("/predict/csv", response_model=PredictResponse, tags=["predict"])
async def predict_csv(file: UploadFile = File(..., description="CSV with headers, or a NASA-format .txt file"),
                      model: str = "auto", create_work_orders: bool = False, fleet: str = "API-UPLOAD") -> PredictResponse:
    try:
        raw = inference.read_upload(await file.read(), file.filename or "upload.csv")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _predict(raw, model, create_work_orders, fleet)


@app.get("/fleets/{subset}", tags=["fleet"])
def fleet(subset: str, severity: str | None = None) -> dict:
    """Latest scored state of a benchmark fleet (written by `fleetguard score`)."""
    path = CFG.path("outputs") / subset.upper() / "fleet_latest.csv"
    if not path.exists():
        raise HTTPException(404, f"No scored fleet for {subset}. Run `fleetguard score --subset {subset}`.")
    df = pd.read_csv(path)
    counts = df["severity"].value_counts().to_dict()
    if severity:
        df = df[df["severity"] == severity.upper()]
    df = df.replace({np.nan: None})
    return {"subset": subset.upper(), "engines": len(df), "severity_counts": counts, "fleet": df.to_dict("records")}


@app.get("/work-orders", tags=["work orders"])
def list_work_orders(fleet: str | None = None, status: str | None = None, severity: str | None = None,
                     limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    orders = [o for o in _store().orders
              if (not fleet or o["subset"] == fleet.upper() or o["subset"] == fleet)
              and (not status or o["status"] == status.upper())
              and (not severity or o["severity"] == severity.upper())]
    keys = ("wo_id", "subset", "engine_id", "status", "severity", "priority", "type", "title", "rul_pred",
            "health_score", "failure_prob", "due_within_cycles", "updated_at")
    return [{k: o.get(k) for k in keys} for o in orders[-limit:]]


@app.get("/work-orders/{wo_id}", tags=["work orders"])
def get_work_order(wo_id: str, markdown: bool = False) -> dict:
    wo = _store().get(wo_id)
    if wo is None:
        raise HTTPException(404, f"Work order {wo_id} not found")
    return {**wo, "markdown": render_markdown(wo)} if markdown else wo


@app.patch("/work-orders/{wo_id}", tags=["work orders"])
def update_work_order(wo_id: str, update: StatusUpdate) -> dict:
    store = _store()
    if store.get(wo_id) is None:
        raise HTTPException(404, f"Work order {wo_id} not found")
    wo = store.set_status(wo_id, update.status, update.note, by=update.by)
    store.save()
    return {"wo_id": wo_id, "status": wo["status"], "history": wo["history"][-3:]}
