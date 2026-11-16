"""Reading the raw C-MAPSS text files and building RUL labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ID_COLS = ["engine_id", "cycle"]
RAW_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS

# Physical meaning of each sensor (Saxena et al., 2008, Table 2).
SENSOR_INFO = {
    "sensor_1": ("T2", "Total temperature at fan inlet", "Fan"),
    "sensor_2": ("T24", "Total temperature at LPC outlet", "Low-pressure compressor"),
    "sensor_3": ("T30", "Total temperature at HPC outlet", "High-pressure compressor"),
    "sensor_4": ("T50", "Total temperature at LPT outlet", "Low-pressure turbine"),
    "sensor_5": ("P2", "Pressure at fan inlet", "Fan"),
    "sensor_6": ("P15", "Total pressure in bypass duct", "Bypass duct"),
    "sensor_7": ("P30", "Total pressure at HPC outlet", "High-pressure compressor"),
    "sensor_8": ("Nf", "Physical fan speed", "Fan"),
    "sensor_9": ("Nc", "Physical core speed", "Core / HP spool"),
    "sensor_10": ("epr", "Engine pressure ratio", "Core / HP spool"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "High-pressure compressor"),
    "sensor_12": ("phi", "Fuel flow to Ps30 ratio", "Combustor / fuel system"),
    "sensor_13": ("NRf", "Corrected fan speed", "Fan"),
    "sensor_14": ("NRc", "Corrected core speed", "Core / HP spool"),
    "sensor_15": ("BPR", "Bypass ratio", "Fan"),
    "sensor_16": ("farB", "Burner fuel-air ratio", "Combustor / fuel system"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "Bleed system"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "Fan"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "Fan"),
    "sensor_20": ("W31", "HPT coolant bleed", "High-pressure turbine"),
    "sensor_21": ("W32", "LPT coolant bleed", "Low-pressure turbine"),
}


def sensor_label(sensor: str) -> str:
    short, desc, _ = SENSOR_INFO.get(sensor, (sensor, sensor, ""))
    return f"{short} ({desc})"


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS file into a typed dataframe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # trailing spaces create empty columns
    if df.shape[1] != len(RAW_COLS):
        raise ValueError(f"{path.name}: expected {len(RAW_COLS)} columns, found {df.shape[1]}")
    df.columns = RAW_COLS
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_subset(raw_dir: str | Path, subset: str) -> dict[str, pd.DataFrame]:
    """Load train, test and the true test RUL for one subset."""
    raw_dir = Path(raw_dir)
    subset = subset.upper()
    train = read_cmapss_file(raw_dir / f"train_{subset}.txt")
    test = read_cmapss_file(raw_dir / f"test_{subset}.txt")
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None)
    rul = rul.dropna(axis=1, how="all")
    test_ids = sorted(test["engine_id"].unique())
    if len(rul) != len(test_ids):
        raise ValueError(f"RUL_{subset}: {len(rul)} labels for {len(test_ids)} test engines")
    test_rul = pd.DataFrame({"engine_id": test_ids, "true_rul": rul.iloc[:, 0].astype(int).values})
    return {"train": train, "test": test, "test_rul": test_rul}


def add_train_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Run-to-failure data: RUL = last cycle - current cycle (optionally capped)."""
    out = df.copy()
    last = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).astype(int)
    if cap is not None:
        out["rul_capped"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(test: pd.DataFrame, test_rul: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Truncated test trajectories: RUL at cycle t = true RUL at last cycle + (last - t)."""
    out = test.merge(test_rul, on="engine_id", how="left", validate="many_to_one")
    last = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (out["true_rul"] + last - out["cycle"]).astype(int)
    out = out.drop(columns="true_rul")
    if cap is not None:
        out["rul_capped"] = out["rul"].clip(upper=cap)
    return out


def load_incoming(incoming_dir: str | Path, subset: str) -> list[pd.DataFrame]:
    """New run-to-failure batches dropped in data/incoming/<subset>_*.txt or .csv."""
    incoming_dir = Path(incoming_dir)
    batches = []
    for path in sorted(incoming_dir.glob(f"{subset.upper()}_*")):
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            missing = [c for c in RAW_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{path.name} is missing columns {missing}")
            df = df[RAW_COLS]
        else:
            df = read_cmapss_file(path)
        df.attrs["source"] = path.name
        batches.append(df)
    return batches


def merge_batches(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append a batch, re-numbering its engines so IDs never collide."""
    new = new.copy()
    offset = int(base["engine_id"].max()) if len(base) else 0
    mapping = {old: offset + i + 1 for i, old in enumerate(sorted(new["engine_id"].unique()))}
    new["engine_id"] = new["engine_id"].map(mapping).astype(int)
    return pd.concat([base, new], ignore_index=True)


def split_engines(engine_ids, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split engine IDs (not rows) so no engine leaks across the split."""
    ids = np.array(sorted(set(engine_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_second = max(1, int(round(len(ids) * fraction)))
    return np.sort(ids[n_second:]), np.sort(ids[:n_second])
