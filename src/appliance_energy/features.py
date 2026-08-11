"""Feature engineering.

Three groups of features are built here:

* time features derived from the timestamp (known arbitrarily far ahead);
* lag and rolling features derived from past values of the target;
* sensor and weather covariates taken from the dataset itself.

Two design matrices are produced:

``make_direct_table``
    The operational design. Every row is a (forecast origin, horizon) pair and
    every feature is measurable at the forecast origin, so the model can be
    used to produce a genuine 24-hour-ahead forecast.

``make_one_step_table``
    The one-step-ahead design used in the demo pipeline, kept so that the
    report can quantify how optimistic it is when it is evaluated as though it
    were a 24-hour-ahead forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------
# Time features
# ------------------------------------------------------------------

def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived only from the timestamp."""
    out = frame.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def time_feature_frame(index: pd.DatetimeIndex, columns=None) -> pd.DataFrame:
    """Calendar features for an arbitrary index, e.g. future timestamps."""
    frame = add_time_features(pd.DataFrame(index=index))
    if columns is not None:
        frame = frame[list(columns)]
    return frame


# ------------------------------------------------------------------
# Sensor groups
# ------------------------------------------------------------------

def available(frame: pd.DataFrame, columns) -> list:
    """Return the subset of ``columns`` that is actually in ``frame``."""
    return [column for column in columns if column in frame.columns]


def sensor_columns(frame: pd.DataFrame) -> list:
    """Indoor temperature and humidity columns plus the lights channel."""
    columns = available(frame, config.INDOOR_TEMP_COLS)
    columns += available(frame, config.INDOOR_HUMIDITY_COLS)
    columns += available(frame, ["lights"])
    return columns


# ------------------------------------------------------------------
# One-step design (as in the demo pipeline)
# ------------------------------------------------------------------

def make_one_step_table(frame: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Supervised table in which every feature is lagged by at least one hour.

    This is the design in the demo pipeline. It is a correct *one-step-ahead*
    design: no feature uses the current or a future value of the target. It is
    not, however, a valid 24-hour-ahead design, because ``lag_1`` would not be
    observed at a forecast origin 24 hours earlier.
    """
    out = add_time_features(frame)

    for lag in config.LAGS_ONE_STEP:
        out[f"lag_{lag}"] = out[target].shift(lag)

    for window in config.ROLLING_WINDOWS:
        shifted = out[target].shift(1)
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()

    return out.dropna()


# ------------------------------------------------------------------
# Direct multi-horizon design (operational)
# ------------------------------------------------------------------

def make_origin_features(
    frame: pd.DataFrame,
    target: str = config.TARGET,
    lags=(0, 1, 2, 3, 6, 12, 24, 48, 168),
    windows=config.ROLLING_WINDOWS,
    include_sensors: bool = True,
) -> pd.DataFrame:
    """Features measurable *at* a forecast origin.

    Row ``t`` contains only quantities that have already been observed by the
    end of hour ``t``: lagged values of the target (``lag_0`` is the value at
    the origin itself), rolling summaries ending at the origin, and the latest
    indoor sensor and outdoor weather readings.
    """
    y = frame[target]
    out = pd.DataFrame(index=frame.index)

    for lag in lags:
        out[f"origin_lag_{lag}"] = y.shift(lag)

    for window in windows:
        out[f"origin_roll_mean_{window}"] = y.rolling(window).mean()
        out[f"origin_roll_std_{window}"] = y.rolling(window).std()

    # Difference features summarising the recent trajectory.
    out["origin_diff_1"] = y.diff(1)
    out["origin_diff_24"] = y.diff(24)

    if include_sensors:
        for column in sensor_columns(frame) + available(frame, config.WEATHER_COLS):
            out[f"origin_{column}"] = frame[column]

    return out


def make_direct_table(
    frame: pd.DataFrame,
    horizon: int = config.HORIZON,
    target: str = config.TARGET,
    include_sensors: bool = True,
    include_future_weather: bool = False,
) -> pd.DataFrame:
    """Build the (origin, horizon) design matrix for direct 24-hour forecasting.

    Parameters
    ----------
    include_future_weather:
        If ``True``, the realised weather at the *target* time is added. Those
        values are not known at the forecast origin in an operational setting,
        so the resulting forecast is a conditional forecast and is reported
        separately in the results.

    Returns a frame with one row per (origin, horizon) pair, columns
    ``origin``, ``target_time``, ``h``, the origin features, the calendar
    features of the target time, and the column ``y`` to be predicted.
    """
    origin_features = make_origin_features(
        frame, target=target, include_sensors=include_sensors
    )

    calendar = time_feature_frame(
        frame.index,
        columns=["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "hour", "dayofweek"],
    )

    weather_cols = available(frame, config.WEATHER_COLS)

    blocks = []

    for h in range(1, horizon + 1):
        block = origin_features.copy()
        block["h"] = h
        block["origin"] = frame.index
        block["target_time"] = frame.index + pd.Timedelta(hours=h)

        # Calendar features of the target time: shift the calendar frame back
        # by h so that row t carries the calendar of t + h.
        future_calendar = calendar.shift(-h)
        for column in calendar.columns:
            block[f"target_{column}"] = future_calendar[column]

        if include_future_weather and weather_cols:
            future_weather = frame[weather_cols].shift(-h)
            for column in weather_cols:
                block[f"future_{column}"] = future_weather[column]

        block["y"] = frame[target].shift(-h)
        blocks.append(block)

    table = pd.concat(blocks, axis=0, ignore_index=True)
    table = table.dropna()
    table = table.sort_values(["origin", "h"]).reset_index(drop=True)

    return table


def direct_feature_columns(table: pd.DataFrame) -> list:
    """Feature columns of a direct design matrix (everything except bookkeeping)."""
    exclude = {"y", "origin", "target_time"}
    return [column for column in table.columns if column not in exclude]


def build_feature_dataset(frame: pd.DataFrame, save_path=None) -> pd.DataFrame:
    """Convenience wrapper used by ``scripts/make_features.py``."""
    table = make_one_step_table(frame)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path)
    return table
