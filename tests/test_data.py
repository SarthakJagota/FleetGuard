import numpy as np
import pandas as pd
import pytest

from fleetguard.config import SUBSETS
from fleetguard.data.cleaning import inject_faults, validate_and_repair
from fleetguard.data.features import FeaturePipeline, last_rows
from fleetguard.data.loader import (
    RAW_COLS, add_test_rul, load_subset, merge_batches, split_engines,
)


@pytest.mark.parametrize("subset,engines", [("FD001", 100), ("FD002", 260), ("FD003", 100), ("FD004", 249)])
def test_load_all_subsets(cfg, subset, engines):
    d = load_subset(cfg.path("raw_data"), subset)
    assert list(d["train"].columns) == RAW_COLS
    assert d["train"]["engine_id"].nunique() == engines
    assert len(d["test_rul"]) == d["test"]["engine_id"].nunique()
    assert not d["train"].isna().any().any()


def test_train_rul_labels(small_train):
    g = small_train.groupby("engine_id")
    assert (g["rul"].min() == 0).all()  # run-to-failure: last cycle has RUL 0
    assert (small_train["rul_capped"] <= 125).all()
    first = small_train.loc[g["cycle"].idxmin()]
    assert (first["rul"] == g["cycle"].max().values - 1).all()


def test_test_rul_labels(fd001):
    lab = add_test_rul(fd001["test"], fd001["test_rul"])
    last = last_rows(lab)
    assert (last["rul"].values == fd001["test_rul"]["true_rul"].values).all()


def test_split_engines_is_disjoint():
    a, b = split_engines(range(1, 101), 0.2, seed=1)
    assert len(set(a) & set(b)) == 0 and len(a) + len(b) == 100 and len(b) == 20


def test_merge_batches_renumbers(fd001):
    base = fd001["train"][fd001["train"]["engine_id"] <= 10]
    new = fd001["train"][fd001["train"]["engine_id"] <= 3]
    merged = merge_batches(base, new)
    assert merged["engine_id"].nunique() == 13


def test_cleaning_repairs_corrupted_feed(small_train):
    dirty = inject_faults(small_train, missing_rate=0.02, duplicate_rate=0.02, seed=3)
    assert dirty[RAW_COLS[2:]].isna().any().any()
    clean, report = validate_and_repair(dirty)
    assert report.dropped_duplicates > 0
    assert report.imputed_values > 0
    assert not clean[RAW_COLS].isna().any().any()
    assert len(clean) == len(small_train)
    assert clean.groupby("engine_id")["cycle"].apply(lambda c: c.is_monotonic_increasing).all()


def test_cleaning_rejects_missing_columns(small_train):
    with pytest.raises(ValueError):
        validate_and_repair(small_train.drop(columns=["sensor_4"]))


@pytest.fixture(scope="module")
def fitted(small_train, cfg):
    return FeaturePipeline.from_config(cfg, "FD001").fit(small_train)


def test_features_drop_constant_sensors(fitted):
    # FD001 sensors 1, 5, 10, 16, 18, 19 are constant (Saxena et al.)
    for s in ("sensor_1", "sensor_5", "sensor_18", "sensor_19"):
        assert s not in fitted.sensors_
    assert {"sensor_4", "sensor_11", "sensor_12"} <= set(fitted.sensors_)


def test_features_are_causal(fitted, small_train):
    """Prediction-time features for a truncated history must equal the full-history row."""
    full = fitted.transform(small_train)
    eng = small_train[small_train["engine_id"] == 5]
    cut = eng[eng["cycle"] <= 60]
    part = fitted.transform(cut)
    cols = fitted.feature_names_
    a = full[(full["engine_id"] == 5) & (full["cycle"] == 60)][cols].to_numpy()
    b = part[part["cycle"] == 60][cols].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_rolling_slope_matches_polyfit(fitted, small_train):
    f = fitted.transform(small_train)
    eng = f[f["engine_id"] == 2].reset_index(drop=True)
    s = fitted.sensors_[0]
    z = eng[f"{s}_mean10"]  # sanity: means exist
    assert z.notna().all()
    raw_z = fitted._normalise(small_train[small_train["engine_id"] == 2].assign(condition=0))[s].to_numpy()
    i = 50
    expected = np.polyfit(np.arange(10), raw_z[i - 9:i + 1], 1)[0]
    assert eng.loc[i, f"{s}_slope10"] == pytest.approx(expected, rel=1e-5, abs=1e-8)


def test_features_have_no_nans(fitted, small_train):
    f = fitted.transform(small_train)
    assert not f[fitted.feature_names_].isna().any().any()


def test_multi_condition_normalisation(cfg):
    d = load_subset(cfg.path("raw_data"), "FD002")
    train = d["train"][d["train"]["engine_id"] <= 40]
    fp = FeaturePipeline.from_config(cfg, "FD002").fit(train)
    assert len(np.unique(fp._conditions(train))) == 6
    z = fp._normalise(train.assign(condition=fp._conditions(train)))[fp.sensors_]
    # Within-regime normalisation removes the regime jumps: every sensor ~N(0,1).
    assert z.mean().abs().max() < 0.05
    # Sensors constant in some regime (e.g. farB) are set to 0 there, so check the rest.
    varies_everywhere = [s for s in fp.sensors_ if fp.cond_std_[s].notna().all()]
    std = z[varies_everywhere].std()
    assert ((std > 0.9) & (std < 1.1)).all()
