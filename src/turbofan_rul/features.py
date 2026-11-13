"""Feature engineering on the sensor time series.

Every feature only looks backwards in time within one engine, so the same
code works for training data and for a live API request with just the
last few cycles of readings.
"""

import numpy as np
import pandas as pd

from .config import LONG_WINDOW, SENSOR_COLS, SHORT_WINDOW


def find_constant_sensors(train: pd.DataFrame, max_unique: int = 2) -> list[str]:
    """Sensors that barely move in training carry no wear signal, so we drop them.

    In FD001 a few sensors are flat or just flip between two values
    (s6 only ever reads 21.60 or 21.61), so "2 distinct values or fewer"
    is a simple, readable rule.
    """
    return [s for s in SENSOR_COLS if train[s].nunique() <= max_unique]


def useful_sensors(train: pd.DataFrame) -> list[str]:
    dropped = set(find_constant_sensors(train))
    return [s for s in SENSOR_COLS if s not in dropped]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of the last `window` points (how fast a sensor is drifting)."""

    def slope(values: np.ndarray) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x -= x.mean()
        return float(np.dot(x, values - values.mean()) / np.dot(x, x))

    return series.rolling(window, min_periods=1).apply(slope, raw=True)


def build_features(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Turn raw readings into one feature row per (engine, cycle).

    For each useful sensor we keep the current value plus a short and a long
    rolling mean, a short rolling std and the short-window slope.
    """
    df = df.sort_values(["unit", "cycle"])
    grouped = df.groupby("unit", sort=False)
    parts = {"cycle": df["cycle"]}
    for s in sensors:
        g = grouped[s]
        parts[s] = df[s]
        parts[f"{s}_mean{SHORT_WINDOW}"] = g.transform(lambda x: x.rolling(SHORT_WINDOW, min_periods=1).mean())
        parts[f"{s}_mean{LONG_WINDOW}"] = g.transform(lambda x: x.rolling(LONG_WINDOW, min_periods=1).mean())
        parts[f"{s}_std{SHORT_WINDOW}"] = g.transform(
            lambda x: x.rolling(SHORT_WINDOW, min_periods=2).std()
        ).fillna(0.0)
        parts[f"{s}_slope{SHORT_WINDOW}"] = g.transform(lambda x: _rolling_slope(x, SHORT_WINDOW))
    return pd.DataFrame(parts, index=df.index)


def feature_columns(sensors: list[str]) -> list[str]:
    cols = ["cycle"]
    for s in sensors:
        cols += [s, f"{s}_mean{SHORT_WINDOW}", f"{s}_mean{LONG_WINDOW}", f"{s}_std{SHORT_WINDOW}", f"{s}_slope{SHORT_WINDOW}"]
    return cols


def sensor_of(feature: str) -> str:
    """Map a feature name back to the sensor it came from ('s11_mean10' -> 's11')."""
    return feature.split("_")[0]


def last_rows(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Keep only each engine's latest feature row (what we predict on at test time).

    `df` is the raw frame the features were built from; rows come back
    ordered by engine number.
    """
    last_idx = df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).index
    return feats.loc[last_idx]
