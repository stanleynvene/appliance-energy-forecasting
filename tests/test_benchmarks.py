"""Tests for the benchmark forecasts, the data preparation and the pipeline design."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import config, data, pipeline
from appliance_energy.models import benchmarks, foundation


@pytest.fixture
def hourly_series():
    index = pd.date_range("2016-01-01", periods=24 * 40, freq="h")
    values = (
        100
        + 40 * np.sin(2 * np.pi * index.hour / 24)
        + 15 * (index.dayofweek >= 5)
        + np.arange(len(index)) * 0.01
    )
    return pd.Series(values, index=index, name=config.TARGET)


@pytest.fixture
def split_series(hourly_series):
    train = hourly_series.iloc[:-config.TEST_STEPS]
    test = hourly_series.iloc[-config.TEST_STEPS:]
    return train, test


def test_forecast_lengths_match_the_test_period(split_series):
    train, test = split_series
    horizon = len(test)

    for name, forecast in benchmarks.all_benchmarks(train, horizon, test.index).items():
        assert len(forecast) == horizon, name
        assert forecast.index.equals(test.index), name
        assert forecast.notna().all(), name


def test_naive_forecast_is_flat_at_the_last_observation(split_series):
    train, test = split_series
    forecast = benchmarks.naive_forecast(train, len(test), test.index)

    assert forecast.nunique() == 1
    assert forecast.iloc[0] == pytest.approx(train.iloc[-1])


def test_daily_seasonal_naive_repeats_the_previous_day(split_series):
    train, test = split_series
    forecast = benchmarks.seasonal_naive_forecast(train, 24, test.index[:24], seasonality=24)

    assert np.allclose(forecast.to_numpy(), train.iloc[-24:].to_numpy())


def test_weekly_seasonal_naive_repeats_the_previous_week(split_series):
    train, test = split_series
    forecast = benchmarks.seasonal_naive_forecast(train, 168, test.index[:168], seasonality=168)

    assert np.allclose(forecast.to_numpy(), train.iloc[-168:].to_numpy())


def test_seasonal_naive_recycles_beyond_one_season(split_series):
    train, test = split_series
    horizon = 48
    forecast = benchmarks.seasonal_naive_forecast(train, horizon, test.index[:horizon], 24)

    assert np.allclose(forecast.to_numpy()[:24], forecast.to_numpy()[24:])


def test_drift_forecast_follows_the_average_slope(split_series):
    train, test = split_series
    forecast = benchmarks.drift_forecast(train, 24, test.index[:24])

    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    assert forecast.diff().dropna().to_numpy() == pytest.approx(slope)


def test_benchmarks_never_touch_the_test_data(hourly_series):
    """Corrupting the test period must not change any benchmark forecast."""
    train = hourly_series.iloc[:-config.TEST_STEPS]
    test = hourly_series.iloc[-config.TEST_STEPS:]

    original = benchmarks.all_benchmarks(train, len(test), test.index)

    corrupted_series = hourly_series.copy()
    corrupted_series.iloc[-config.TEST_STEPS:] = -999.0
    corrupted_train = corrupted_series.iloc[:-config.TEST_STEPS]

    corrupted = benchmarks.all_benchmarks(corrupted_train, len(test), test.index)

    for name in original:
        assert np.allclose(original[name].to_numpy(), corrupted[name].to_numpy()), name


def test_rolling_origins_are_one_day_apart(hourly_series):
    origins = pipeline.rolling_origins(hourly_series, n_origins=14, horizon=24)

    assert len(origins) == 14
    assert (origins.to_series().diff().dropna() == pd.Timedelta(hours=24)).all()
    assert origins[-1] == hourly_series.index[-25]


def test_rolling_benchmarks_cover_the_whole_test_period(hourly_series):
    forecasts = pipeline.rolling_benchmarks(hourly_series, n_origins=14, horizon=24)
    test_index = hourly_series.index[-336:]

    for name, forecast in forecasts.items():
        assert forecast.index.equals(test_index), name
        assert forecast.notna().all(), name


def test_train_test_split_is_chronological_and_the_right_length(hourly_series):
    train, test = data.train_test_split(hourly_series, test_steps=config.TEST_STEPS)

    assert len(test) == config.TEST_STEPS
    assert len(train) + len(test) == len(hourly_series)
    assert train.index.max() < test.index.min()


def test_train_test_split_rejects_a_too_short_series():
    with pytest.raises(ValueError):
        data.train_test_split(pd.Series(np.arange(10)), test_steps=336)


def test_processed_hourly_data_has_a_regular_index_and_no_missing_target():
    """The real processed dataset, if it has been built, must be clean."""
    if not config.HOURLY_CSV.exists():
        pytest.skip("processed dataset not built yet")

    hourly = data.load_hourly()

    assert hourly[config.TARGET].notna().all()
    assert hourly.index.is_monotonic_increasing
    assert not hourly.index.has_duplicates
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_foundation_fallback_returns_ordered_quantiles(hourly_series):
    index = hourly_series.index[-24:]
    history = hourly_series.iloc[:-24]

    median, lower, upper, meta = foundation.forecast_foundation(
        history, horizon=24, index=index, backend="fallback"
    )

    assert meta["used_fallback"] is True
    assert len(median) == 24
    assert (lower <= median).all()
    assert (median <= upper).all()


def test_foundation_fallback_uses_only_the_history(hourly_series):
    index = hourly_series.index[-24:]
    history = hourly_series.iloc[:-24]

    first, _, _, _ = foundation.forecast_foundation(history, 24, index, backend="fallback")

    corrupted = hourly_series.copy()
    corrupted.iloc[-24:] = -999.0
    second, _, _, _ = foundation.forecast_foundation(
        corrupted.iloc[:-24], 24, index, backend="fallback"
    )

    assert np.allclose(first.to_numpy(), second.to_numpy())
