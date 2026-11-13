"""Tests for loading, labelling, feature building and the train/test split."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_engine, needs_data
from sklearn.model_selection import GroupKFold

from turbofan_rul.config import RAW_COLS, RUL_CAP
from turbofan_rul.data import add_rul, read_raw
from turbofan_rul.features import build_features, feature_columns, find_constant_sensors, last_rows
from turbofan_rul.model import nasa_score, risk_level, rmse


# ---- labels ----------------------------------------------------------------
def test_rul_counts_down_to_zero_at_failure():
    df = add_rul(make_engine(10), cap=None)
    assert df["rul"].tolist() == list(range(9, -1, -1))


def test_rul_is_capped():
    df = add_rul(make_engine(200))
    assert df["rul"].max() == RUL_CAP
    assert df["rul"].iloc[-1] == 0


def test_rul_is_per_engine():
    df = pd.concat([make_engine(5, unit=1), make_engine(8, unit=2)])
    df = add_rul(df, cap=None)
    assert df.groupby("unit")["rul"].max().to_dict() == {1: 4, 2: 7}


# ---- raw files -----------------------------------------------------------------
def test_read_raw_handles_trailing_spaces(tmp_path):
    row = " ".join(["1", "1"] + ["0.5"] * 24) + "  \n"
    path = tmp_path / "train_X.txt"
    path.write_text(row * 3)
    df = read_raw(path)
    assert list(df.columns) == RAW_COLS
    assert len(df) == 3


def test_read_raw_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        read_raw(path)


@needs_data
def test_fd001_shapes(train_df, test_data):
    test, truth = test_data
    assert train_df["unit"].nunique() == 100 and len(train_df) == 20631
    assert test["unit"].nunique() == 100 and len(test) == 13096
    assert len(truth) == 100


@needs_data
def test_constant_sensors_found(train_df):
    assert find_constant_sensors(train_df) == ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


# ---- features ----------------------------------------------------------------
def test_features_only_look_backwards():
    """Changing a future reading must not change features at earlier cycles."""
    engine = make_engine(40)
    changed = engine.copy()
    changed.loc[changed["cycle"] == 40, "s2"] += 50
    a = build_features(engine, ["s2"])
    b = build_features(changed, ["s2"])
    pd.testing.assert_frame_equal(a.iloc[:39], b.iloc[:39])
    assert not a.iloc[39].equals(b.iloc[39])


def test_features_do_not_leak_between_engines():
    one = make_engine(20, unit=1, seed=1)
    two = make_engine(20, unit=2, seed=2)
    together = build_features(pd.concat([one, two], ignore_index=True), ["s3"])
    alone = build_features(two.reset_index(drop=True), ["s3"])
    np.testing.assert_allclose(together.iloc[20:].to_numpy(), alone.to_numpy())


def test_slope_feature_picks_up_drift():
    engine = make_engine(30)
    engine["s4"] = 2.0 * engine["cycle"]  # perfectly straight line, slope 2
    feats = build_features(engine, ["s4"])
    assert feats["s4_slope10"].iloc[-1] == pytest.approx(2.0)


def test_single_cycle_still_gives_features():
    feats = build_features(make_engine(1), ["s2", "s3"])
    assert list(feats.columns) == feature_columns(["s2", "s3"])
    assert not feats.isna().any().any()


def test_last_rows_takes_latest_cycle_per_engine():
    df = pd.concat([make_engine(5, unit=2), make_engine(3, unit=1)], ignore_index=True)
    feats = build_features(df, ["s2"])
    last = last_rows(df, feats)
    assert last["cycle"].tolist() == [3, 5]


# ---- split -------------------------------------------------------------------
@needs_data
def test_cv_split_keeps_engines_apart(train_df):
    for tr, va in GroupKFold(n_splits=5).split(train_df, groups=train_df["unit"]):
        assert set(train_df["unit"].iloc[tr]).isdisjoint(train_df["unit"].iloc[va])


# ---- metrics -----------------------------------------------------------------
def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_nasa_score_punishes_late_more_than_early():
    late = nasa_score([50], [60])
    early = nasa_score([50], [40])
    assert late == pytest.approx(np.exp(1) - 1)
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late > early
    assert nasa_score([50], [50]) == 0


@pytest.mark.parametrize("rul,level", [(0, "high"), (30, "high"), (30.1, "medium"), (75, "medium"), (76, "low")])
def test_risk_levels(rul, level):
    assert risk_level(rul) == level
