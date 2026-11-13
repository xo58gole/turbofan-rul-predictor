"""Shared test fixtures.

The API tests use a small, quickly-trained model so they don't depend on
the full training run having happened first.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from turbofan_rul.config import DATA_DIR, SENSOR_COLS, SETTING_COLS
from turbofan_rul.data import load_test, load_train
from turbofan_rul.features import build_features, useful_sensors
from turbofan_rul.model import RULModel

HAS_DATA = (DATA_DIR / "train_FD001.txt").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run scripts/download_data.py first")


def make_engine(n_cycles: int, unit: int = 1, seed: int = 0) -> pd.DataFrame:
    """A fake engine: every sensor drifts upward a little each cycle, plus noise."""
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    data = {"unit": unit, "cycle": cycles}
    for c in SETTING_COLS:
        data[c] = rng.normal(0, 0.001, n_cycles)
    for i, s in enumerate(SENSOR_COLS):
        data[s] = 100 + i + 0.05 * cycles + rng.normal(0, 0.1, n_cycles)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def train_df():
    if not HAS_DATA:
        pytest.skip("run scripts/download_data.py first")
    return load_train()


@pytest.fixture(scope="session")
def test_data():
    if not HAS_DATA:
        pytest.skip("run scripts/download_data.py first")
    return load_test()


@pytest.fixture(scope="session")
def small_model(train_df) -> RULModel:
    sensors = useful_sensors(train_df)
    feats = build_features(train_df, sensors)
    model = RULModel(
        estimator=HistGradientBoostingRegressor(max_iter=40, random_state=0),
        sensors=sensors,
        healthy_reference=feats.median().to_dict(),
        name="test_model",
    )
    model.estimator.fit(feats[model.feature_cols], train_df.loc[feats.index, "rul"])
    return model
