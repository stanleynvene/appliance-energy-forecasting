"""Time-series foundation model.

Primary model: **Chronos-2** (``amazon/chronos-2``), used zero-shot. Nothing is
fitted to the appliance data: the pre-trained network is given the historical
target series as context and asked for a 24-hour predictive distribution. A
Chronos-Bolt variant is also provided, matching the tutorial.

Offline fallback
----------------
Chronos downloads its weights from the Hugging Face hub the first time it runs.
If ``chronos-forecasting``/``torch`` are not installed, or the hub cannot be
reached, ``forecast_foundation`` falls back to an explicitly documented
statistical model (:func:`seasonal_quantile_forecast`) and records
``used_fallback=True`` in the returned metadata. The pipeline propagates that
flag to ``outputs/metrics/foundation_model_backend.json`` and the results
tables label the row accordingly, so a fallback run is never silently reported
as a Chronos result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

CHRONOS_2_MODEL = "amazon/chronos-2"
CHRONOS_BOLT_MODEL = "amazon/chronos-bolt-small"

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


# ------------------------------------------------------------------
# Chronos-2
# ------------------------------------------------------------------

def make_context_frame(history: pd.Series, item_id: str = "appliances") -> pd.DataFrame:
    """Chronos-2 expects a long frame with an id, a timestamp and a target."""
    return pd.DataFrame(
        {
            "id": item_id,
            "timestamp": history.index,
            "target": history.to_numpy(dtype=float),
        }
    )


def load_chronos2_pipeline(model_name: str = CHRONOS_2_MODEL):
    """Load the pre-trained Chronos-2 pipeline onto GPU if one is available."""
    import torch
    from chronos import Chronos2Pipeline

    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    return Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)


def get_quantile_column(frame: pd.DataFrame, q: float):
    """Chronos versions name quantile columns slightly differently."""
    for candidate in [q, str(q), f"{q:.1f}", f"{q:.2f}"]:
        if candidate in frame.columns:
            return candidate

    raise KeyError(f"Could not find quantile column for {q}. Columns: {list(frame.columns)}")


def chronos2_forecast(
    pipeline,
    history: pd.Series,
    horizon: int,
    index,
    quantile_levels=DEFAULT_QUANTILES,
):
    """Zero-shot Chronos-2 forecast for a single 24-hour block."""
    context = make_context_frame(history)

    prediction = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    prediction = prediction.sort_values("timestamp").tail(horizon)

    q_low, q_mid, q_high = (get_quantile_column(prediction, q) for q in quantile_levels)

    lower = pd.Series(prediction[q_low].to_numpy(dtype=float), index=index)
    median = pd.Series(prediction[q_mid].to_numpy(dtype=float), index=index)
    upper = pd.Series(prediction[q_high].to_numpy(dtype=float), index=index)

    return median, lower, upper


# ------------------------------------------------------------------
# Chronos-Bolt
# ------------------------------------------------------------------

def load_bolt_pipeline(model_name: str = CHRONOS_BOLT_MODEL):
    import torch
    from chronos import BaseChronosPipeline

    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    return BaseChronosPipeline.from_pretrained(model_name, device_map=device_map)


def bolt_forecast(
    pipeline,
    history: pd.Series,
    horizon: int,
    index,
    context_length: int = 24 * 28,
    quantile_levels=DEFAULT_QUANTILES,
):
    """Zero-shot Chronos-Bolt forecast using the most recent context window."""
    import torch

    context = torch.tensor(
        history.iloc[-context_length:].to_numpy(dtype=float), dtype=torch.float32
    ).unsqueeze(0)

    quantiles, _mean = pipeline.predict_quantiles(
        inputs=context,
        prediction_length=horizon,
        quantile_levels=list(quantile_levels),
    )

    values = quantiles[0].detach().cpu().numpy()

    lower = pd.Series(values[:, 0], index=index)
    median = pd.Series(values[:, 1], index=index)
    upper = pd.Series(values[:, 2], index=index)

    return median, lower, upper


# ------------------------------------------------------------------
# Offline fallback
# ------------------------------------------------------------------

def seasonal_quantile_forecast(
    history: pd.Series,
    horizon: int,
    index,
    weekly_period: int = config.WEEKLY_PERIOD,
    daily_period: int = config.DAILY_PERIOD,
    n_weeks: int = 8,
    n_days: int = 14,
    weekly_weight: float = 0.5,
    quantile_levels=DEFAULT_QUANTILES,
):
    """Documented statistical stand-in for the foundation model.

    For each step of the horizon the forecast pools two empirical samples:

    * the observations at the same hour-of-week over the last ``n_weeks`` weeks;
    * the observations at the same hour-of-day over the last ``n_days`` days.

    The median of the pooled sample is the point forecast and the 10th and 90th
    percentiles give an 80% interval. This is a genuine probabilistic
    forecasting method, but it is *not* a foundation model and is reported as a
    fallback wherever it is used.
    """
    values = history.to_numpy(dtype=float)
    n = len(values)

    lows, mids, highs = [], [], []

    for step in range(1, horizon + 1):
        weekly_positions = [n - k * weekly_period + step - 1 for k in range(1, n_weeks + 1)]
        daily_positions = [n - k * daily_period + step - 1 for k in range(1, n_days + 1)]

        weekly_sample = [values[p] for p in weekly_positions if 0 <= p < n]
        daily_sample = [values[p] for p in daily_positions if 0 <= p < n]

        sample = np.array(weekly_sample + daily_sample, dtype=float)
        weights = np.array(
            [weekly_weight] * len(weekly_sample)
            + [1.0 - weekly_weight] * len(daily_sample),
            dtype=float,
        )

        if sample.size == 0:  # pragma: no cover - only for very short histories
            sample = values[-daily_period:]
            weights = np.ones_like(sample)

        order = np.argsort(sample)
        sorted_sample = sample[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        cumulative /= sorted_weights.sum()

        q_low, q_mid, q_high = (
            np.interp(q, cumulative, sorted_sample) for q in quantile_levels
        )

        lows.append(q_low)
        mids.append(q_mid)
        highs.append(q_high)

    return (
        pd.Series(mids, index=index),
        pd.Series(lows, index=index),
        pd.Series(highs, index=index),
    )


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

def forecast_foundation(
    history: pd.Series,
    horizon: int,
    index,
    backend: str = "chronos2",
    pipeline=None,
    quantile_levels=DEFAULT_QUANTILES,
):
    """Forecast one block with the requested backend, falling back if necessary.

    Returns ``(median, lower, upper, metadata)`` where metadata records which
    backend actually produced the numbers.
    """
    metadata = {"requested_backend": backend, "used_fallback": False}

    if backend in ("chronos2", "bolt"):
        try:
            if pipeline is None:
                pipeline = (
                    load_chronos2_pipeline() if backend == "chronos2" else load_bolt_pipeline()
                )

            if backend == "chronos2":
                median, lower, upper = chronos2_forecast(
                    pipeline, history, horizon, index, quantile_levels
                )
                metadata["backend"] = CHRONOS_2_MODEL
            else:
                median, lower, upper = bolt_forecast(
                    pipeline, history, horizon, index, quantile_levels=quantile_levels
                )
                metadata["backend"] = CHRONOS_BOLT_MODEL

            metadata["pipeline"] = pipeline
            return median, lower, upper, metadata

        except Exception as error:  # noqa: BLE001 - any failure means no weights
            metadata["used_fallback"] = True
            metadata["fallback_reason"] = f"{type(error).__name__}: {error}"

    median, lower, upper = seasonal_quantile_forecast(
        history, horizon, index, quantile_levels=quantile_levels
    )
    metadata["backend"] = "seasonal_quantile_fallback"
    metadata.setdefault("fallback_reason", "backend explicitly set to fallback")
    metadata["used_fallback"] = True

    return median, lower, upper, metadata


def rolling_origin_foundation(
    y: pd.Series,
    n_origins: int,
    horizon: int,
    backend: str = "chronos2",
    name: str = "foundation_model",
    quantile_levels=DEFAULT_QUANTILES,
):
    """Rolling-origin backtest of the foundation model.

    At each origin the model is given every observation up to that point as
    context. The weights are never updated, so the model remains zero-shot.
    """
    test_steps = n_origins * horizon
    split = len(y) - test_steps

    medians, lowers, uppers = [], [], []
    pipeline = None
    metadata = {}

    for origin in range(n_origins):
        start = split + origin * horizon
        stop = start + horizon

        history = y.iloc[:start]
        index = y.index[start:stop]

        median, lower, upper, metadata = forecast_foundation(
            history=history,
            horizon=horizon,
            index=index,
            backend=backend,
            pipeline=pipeline,
            quantile_levels=quantile_levels,
        )

        pipeline = metadata.pop("pipeline", None)

        medians.append(median)
        lowers.append(lower)
        uppers.append(upper)

        if metadata.get("used_fallback"):
            backend = "fallback"

    point = pd.concat(medians).rename(name)
    lower = pd.concat(lowers).rename(f"{name}_lower")
    upper = pd.concat(uppers).rename(f"{name}_upper")

    return point, lower, upper, metadata
