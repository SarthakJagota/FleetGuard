import pytest

from fleetguard.config import load_config
from fleetguard.data.loader import add_train_rul, load_subset


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def fd001(cfg):
    return load_subset(cfg.path("raw_data"), "FD001")


@pytest.fixture(scope="session")
def small_train(fd001):
    """First 30 engines of FD001 with RUL labels - enough to fit quickly."""
    df = fd001["train"]
    return add_train_rul(df[df["engine_id"] <= 30].reset_index(drop=True), cap=125)


@pytest.fixture
def tmp_cfg(cfg, tmp_path):
    """Config whose writable paths point into a temp directory."""
    return cfg.override(paths={
        "raw_data": str(cfg.path("raw_data")),
        "incoming_data": str(tmp_path / "incoming"),
        "models": str(tmp_path / "models"),
        "outputs": str(tmp_path / "outputs"),
    })
