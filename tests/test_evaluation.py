"""Tests for the accuracy metrics."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import evaluation


@pytest.fixture
def toy_train():
    index = pd.date_range("2016-01-01", periods=24 * 20, freq="h")
    values = 100 + 30 * np.sin(2 * np.pi * index.hour / 24)
    return pd.Series(values, index=index)


@pytest.fixture
def toy_test(toy_train):
    index = pd.date_range(toy_train.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    values = 100 + 30 * np.sin(2 * np.pi * index.hour / 24)
    return pd.Series(values, index=index)


def test_perfect_forecast_gives_zero_errors(toy_test, toy_train):
    scores = evaluation.evaluate_forecast("perfect", toy_test, toy_test, toy_train)

    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["RMSE"] == pytest.approx(0.0)
    assert scores["Bias"] == pytest.approx(0.0)


def test_mase_is_zero_for_a_perfect_forecast():
    """A perfect forecast must have MASE exactly zero for a non-degenerate series."""
    rng = np.random.default_rng(1)
    index = pd.date_range("2016-01-01", periods=24 * 20, freq="h")
    train = pd.Series(100 + rng.normal(0, 10, len(index)), index=index)

    test_index = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    test = pd.Series(100 + rng.normal(0, 10, 24), index=test_index)

    assert evaluation.mase(test, test, train, seasonality=24) == pytest.approx(0.0)


def test_mase_equals_one_when_out_of_sample_error_matches_the_scale(toy_train):
    """MASE is by construction MAE divided by the in-sample seasonal naive MAE."""
    rng = np.random.default_rng(2)
    train = pd.Series(rng.normal(100, 20, 24 * 30),
                      index=pd.date_range("2016-01-01", periods=24 * 30, freq="h"))
    scale = evaluation.seasonal_naive_scale(train, seasonality=24)

    test = pd.Series(np.full(24, 100.0))
    prediction = test + scale  # every error is exactly the scale

    assert evaluation.mase(test, prediction, train, 24) == pytest.approx(1.0)


def test_bias_sign_is_forecast_minus_actual(toy_test, toy_train):
    too_high = toy_test + 10
    too_low = toy_test - 10

    assert evaluation.bias(toy_test, too_high) == pytest.approx(10.0)
    assert evaluation.bias(toy_test, too_low) == pytest.approx(-10.0)


def test_rmse_is_at_least_mae(toy_test):
    rng = np.random.default_rng(3)
    prediction = toy_test + rng.normal(0, 15, len(toy_test))

    assert evaluation.rmse(toy_test, prediction) >= evaluation.mae(toy_test, prediction)


def test_seasonal_naive_scale_rejects_short_series():
    with pytest.raises(ValueError):
        evaluation.seasonal_naive_scale(pd.Series(np.arange(10)), seasonality=24)


def test_coverage_bounds():
    y = pd.Series(np.arange(10, dtype=float))

    assert evaluation.coverage(y, y - 1, y + 1) == pytest.approx(1.0)
    assert evaluation.coverage(y, y + 1, y + 2) == pytest.approx(0.0)


def test_evaluate_all_returns_one_row_per_model_sorted_by_mase(toy_test, toy_train):
    forecasts = {
        "perfect": toy_test,
        "offset": toy_test + 25,
    }

    results = evaluation.evaluate_all(forecasts, toy_test, toy_train)

    assert list(results["model"]) == ["perfect", "offset"]
    assert set(results.columns) == {"model", "MAE", "RMSE", "MASE", "Bias"}


def test_errors_by_horizon_has_one_entry_per_step(toy_test):
    prediction = toy_test + 5
    by_step = evaluation.errors_by_horizon(toy_test, prediction, horizon=24)

    assert len(by_step) == 24
    assert np.allclose(by_step.to_numpy(), 5.0)
