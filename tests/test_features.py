"""Tests for the feature engineering code, with an emphasis on data leakage."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import config, features


@pytest.fixture
def toy_frame():
    index = pd.date_range("2016-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(0)

    frame = pd.DataFrame(
        {
            config.TARGET: 100
            + 40 * np.sin(2 * np.pi * index.hour / 24)
            + rng.normal(0, 5, len(index)),
            "T_out": rng.normal(5, 2, len(index)),
            "RH_out": rng.normal(80, 5, len(index)),
            "Windspeed": rng.normal(4, 1, len(index)),
            "Visibility": rng.normal(40, 3, len(index)),
            "Tdewpoint": rng.normal(2, 1, len(index)),
            "Press_mm_hg": rng.normal(755, 3, len(index)),
            "lights": rng.integers(0, 30, len(index)).astype(float),
            "T1": rng.normal(20, 1, len(index)),
            "RH_1": rng.normal(40, 2, len(index)),
        },
        index=index,
    )
    return frame


def test_time_features_are_deterministic_functions_of_the_timestamp(toy_frame):
    out = features.add_time_features(toy_frame)

    assert (out["hour"] == out.index.hour).all()
    assert (out["dayofweek"] == out.index.dayofweek).all()
    assert set(out["is_weekend"].unique()) <= {0, 1}
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
    assert np.allclose(out["dow_sin"] ** 2 + out["dow_cos"] ** 2, 1.0)


def test_one_step_lags_never_use_future_target_values(toy_frame):
    table = features.make_one_step_table(toy_frame)
    y = toy_frame[config.TARGET]

    for lag in config.LAGS_ONE_STEP:
        column = f"lag_{lag}"
        expected = y.shift(lag).reindex(table.index)
        assert np.allclose(table[column].to_numpy(), expected.to_numpy())


def test_one_step_rolling_features_are_shifted(toy_frame):
    """A rolling feature must exclude the current observation of the target."""
    table = features.make_one_step_table(toy_frame)
    y = toy_frame[config.TARGET]

    for window in config.ROLLING_WINDOWS:
        expected = y.shift(1).rolling(window).mean().reindex(table.index)
        assert np.allclose(table[f"roll_mean_{window}"].to_numpy(), expected.to_numpy())

    # An unshifted rolling mean would correlate with the target far more
    # strongly than the shifted one; this guards against reintroducing it.
    unshifted = y.rolling(24).mean().reindex(table.index)
    shifted = table["roll_mean_24"]
    assert shifted.corr(table[config.TARGET]) < unshifted.corr(table[config.TARGET])


def test_origin_features_only_use_past_and_present(toy_frame):
    origin = features.make_origin_features(toy_frame)
    y = toy_frame[config.TARGET]

    assert np.allclose(origin["origin_lag_0"].dropna(), y.reindex(origin["origin_lag_0"].dropna().index))

    for lag in (1, 24, 168):
        column = f"origin_lag_{lag}"
        expected = y.shift(lag)
        mask = origin[column].notna()
        assert np.allclose(origin.loc[mask, column], expected.loc[mask])


def test_direct_table_features_are_known_at_the_origin(toy_frame):
    """Every predictor must be measurable at ``origin``; only ``y`` may be later."""
    table = features.make_direct_table(toy_frame, horizon=24)
    y = toy_frame[config.TARGET]

    assert (table["target_time"] - table["origin"] == pd.to_timedelta(table["h"], unit="h")).all()
    assert table["h"].between(1, 24).all()

    # origin_lag_0 is the target value at the origin, never at the target time.
    expected_origin_value = y.reindex(pd.DatetimeIndex(table["origin"])).to_numpy()
    assert np.allclose(table["origin_lag_0"].to_numpy(), expected_origin_value)

    # y is the target value at the target time.
    expected_y = y.reindex(pd.DatetimeIndex(table["target_time"])).to_numpy()
    assert np.allclose(table["y"].to_numpy(), expected_y)

    # No origin feature may equal the value being predicted.
    assert not np.allclose(table["origin_lag_0"].to_numpy(), table["y"].to_numpy())


def test_direct_table_target_calendar_matches_target_time(toy_frame):
    table = features.make_direct_table(toy_frame, horizon=24)
    target_time = pd.DatetimeIndex(table["target_time"])

    assert (table["target_hour"].to_numpy() == target_time.hour).all()
    assert (table["target_dayofweek"].to_numpy() == target_time.dayofweek).all()


def test_future_weather_is_absent_from_the_operational_design(toy_frame):
    operational = features.make_direct_table(toy_frame, include_future_weather=False)
    conditional = features.make_direct_table(toy_frame, include_future_weather=True)

    assert not [c for c in operational.columns if c.startswith("future_")]
    assert [c for c in conditional.columns if c.startswith("future_")]


def test_direct_table_has_no_missing_values(toy_frame):
    table = features.make_direct_table(toy_frame)
    assert not table.isna().any().any()
    assert len(table) > 0
