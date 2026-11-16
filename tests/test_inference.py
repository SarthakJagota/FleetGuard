"""Predicting on new data: uploads, single readings, what-if scenarios."""

import pandas as pd
import pytest

from fleetguard import inference
from fleetguard.agent.work_orders import WorkOrderStore
from fleetguard.data.loader import RAW_COLS


@pytest.fixture(scope="module")
def bundle(fast_bundle):
    return fast_bundle


@pytest.fixture(scope="module")
def engine(fd001):
    return fd001["test"][fd001["test"]["engine_id"] == 31].reset_index(drop=True)


def test_read_upload_accepts_aliases_and_nasa_txt(engine):
    renamed = engine.rename(columns={"engine_id": "unit_number", "cycle": "time_in_cycles", "sensor_4": "s4",
                                     "setting_1": "op_setting_1"})
    df = inference.read_upload(renamed.to_csv(index=False).encode(), "e.csv")
    assert list(df.columns) == RAW_COLS and len(df) == len(engine)
    txt = engine.to_csv(sep=" ", header=False, index=False)
    assert len(inference.read_upload(txt.encode(), "e.txt")) == len(engine)


def test_read_upload_reports_missing_columns(engine):
    with pytest.raises(ValueError, match="sensor_7"):
        inference.read_upload(engine.drop(columns=["sensor_7"]).to_csv(index=False).encode(), "e.csv")


def test_suggest_subset_picks_by_regimes_and_availability(fd001, cfg):
    from fleetguard.data.loader import load_subset

    fd2 = load_subset(cfg.path("raw_data"), "FD002")["test"]
    assert inference.count_regimes(fd001["test"]) == 1 and inference.count_regimes(fd2) == 6
    # Prefer the model trained on both fault modes...
    assert inference.suggest_subset(fd001["test"])[0] == "FD003"
    assert inference.suggest_subset(fd2)[0] == "FD004"
    # ...fall back to the other model of the right regime type...
    assert inference.suggest_subset(fd001["test"], ["FD001", "FD002"])[0] == "FD001"
    assert inference.suggest_subset(fd2, ["FD001", "FD002"])[0] == "FD002"
    # ...and say so when no model of the right type is trained.
    subset, reason = inference.suggest_subset(fd2, ["FD001"])
    assert subset == "FD001" and "unreliable" in reason


def test_upload_creates_work_orders_under_own_label(bundle, fd001, tmp_path):
    store = WorkOrderStore(tmp_path / "wo.json")
    fleet, hist, report = inference.predict_upload(bundle, fd001["test"][fd001["test"]["engine_id"] <= 20], store, "UPLOAD-test")
    assert len(fleet) == 20 and {"rul_pred", "rul_lower", "rul_upper", "severity"} <= set(fleet.columns)
    assert all(o["subset"] == "UPLOAD-test" for o in store.orders)
    assert len(store.orders) == (fleet["severity"] != "LOW").sum()


def test_append_reading_extends_history(bundle, engine):
    new = inference.append_reading(engine, {"sensor_4": 1450.0})
    assert len(new) == len(engine) + 1
    assert new["cycle"].iloc[-1] == engine["cycle"].max() + 1
    assert new["sensor_4"].iloc[-1] == 1450.0
    assert new["sensor_11"].iloc[-1] == engine["sensor_11"].iloc[-1]  # untouched values carried over
    res = inference.latest_prediction(bundle, new)
    assert res["decision"]["cycle"] == new["cycle"].iloc[-1]


def test_single_reading_is_flagged_not_trusted(bundle, engine):
    res = inference.latest_prediction(bundle, engine.head(1))
    assert "INSUFFICIENT_HISTORY" in res["decision"]["reason_codes"]


def test_what_if_degradation_lowers_rul(bundle, engine):
    worse = inference.what_if(bundle, engine, {"sensor_4": 4, "sensor_11": 4, "sensor_12": -4}, last_n=40)
    assert (worse["cycle"] == engine["cycle"]).all()
    assert (worse["sensor_4"].iloc[: len(engine) - 40] == engine["sensor_4"].iloc[: len(engine) - 40]).all()
    cmp = inference.compare(bundle, engine, worse)
    assert cmp["after"]["rul_pred"] < cmp["before"]["rul_pred"]
    same = inference.what_if(bundle, engine, {}, last_n=40)
    pd.testing.assert_frame_equal(same, engine[RAW_COLS].reset_index(drop=True))
