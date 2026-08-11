"""Simple benchmark forecasts.

Every function takes the history available at the forecast origin and returns
a forecast indexed by the timestamps to be predicted, so that all benchmarks
can be re-issued at each origin of the rolling-origin evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def mean_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    """Forecast every future value by the mean of the history."""
    return pd.Series(float(y_train.mean()), index=index, name="mean")


def naive_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    """Forecast every future value by the last observation."""
    return pd.Series(float(y_train.iloc[-1]), index=index, name="naive")


def seasonal_naive_forecast(y_train: pd.Series, horizon: int, index, seasonality: int) -> pd.Series:
    """Repeat the most recent complete season.

    With hourly data, ``seasonality=24`` gives the same hour yesterday and
    ``seasonality=168`` gives the same hour last week. When the horizon is
    longer than one season the forecast recycles its own values, which is the
    standard recursive definition.
    """
    if len(y_train) < seasonality:
        raise ValueError("History is shorter than one seasonal period.")

    history = list(y_train.to_numpy(dtype=float))
    values = []

    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    name = "seasonal_naive" if seasonality not in (24, 168) else (
        "seasonal_naive_daily" if seasonality == 24 else "seasonal_naive_weekly"
    )

    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train: pd.Series, horizon: int, index) -> pd.Series:
    """Extrapolate the straight line through the first and last observations."""
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)

    values = [float(y_train.iloc[-1] + slope * step) for step in range(1, horizon + 1)]

    return pd.Series(values, index=index, name="drift")


def seasonal_mean_profile_forecast(y_train: pd.Series, horizon: int, index,
                                   seasonality: int = 168) -> pd.Series:
    """Average of the same seasonal slot over the whole history.

    Included as a slightly stronger, low-variance alternative to the seasonal
    naive forecast: instead of copying one past week it averages every past
    observation that falls in the same hour-of-week slot.
    """
    history = pd.Series(y_train.to_numpy(dtype=float))
    slot = np.arange(len(history)) % seasonality
    profile = history.groupby(slot).mean()

    start = len(history) % seasonality
    slots = (start + np.arange(horizon)) % seasonality

    return pd.Series(profile.loc[slots].to_numpy(), index=index, name="seasonal_mean_profile")


def all_benchmarks(y_train: pd.Series, horizon: int, index) -> dict:
    """Produce every benchmark forecast required by the brief."""
    return {
        "mean": mean_forecast(y_train, horizon, index),
        "naive": naive_forecast(y_train, horizon, index),
        "seasonal_naive_daily": seasonal_naive_forecast(
            y_train, horizon, index, seasonality=config.DAILY_PERIOD
        ),
        "seasonal_naive_weekly": seasonal_naive_forecast(
            y_train, horizon, index, seasonality=config.WEEKLY_PERIOD
        ),
        "drift": drift_forecast(y_train, horizon, index),
        "seasonal_mean_profile": seasonal_mean_profile_forecast(
            y_train, horizon, index, seasonality=config.WEEKLY_PERIOD
        ),
    }
