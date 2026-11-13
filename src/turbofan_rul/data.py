"""Loading the raw C-MAPSS files and building RUL labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_COLS, RUL_CAP, SUBSET


def read_raw(path: Path) -> pd.DataFrame:
    """Read one space-separated C-MAPSS file into a DataFrame with named columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Some copies of the files have trailing spaces, which show up as empty columns.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(RAW_COLS):
        raise ValueError(f"{path.name}: expected {len(RAW_COLS)} columns, got {df.shape[1]}")
    df.columns = RAW_COLS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Add the RUL label to training data.

    Training engines run until they fail, so RUL at each row is simply
    "last cycle for this engine minus the current cycle".
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last_cycle - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def load_train(data_dir: Path = DATA_DIR, subset: str = SUBSET) -> pd.DataFrame:
    return add_rul(read_raw(data_dir / f"train_{subset}.txt"))


def load_test(data_dir: Path = DATA_DIR, subset: str = SUBSET) -> tuple[pd.DataFrame, np.ndarray]:
    """Test engines stop some time before failure. The true RUL at their last
    recorded cycle comes in a separate file, one line per engine."""
    test = read_raw(data_dir / f"test_{subset}.txt")
    truth = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None)
    truth = truth.dropna(axis=1, how="all").iloc[:, 0].to_numpy(dtype=float)
    if len(truth) != test["unit"].nunique():
        raise ValueError("RUL file length doesn't match the number of test engines")
    return test, truth
