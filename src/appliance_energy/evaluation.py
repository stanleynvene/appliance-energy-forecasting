"""Forecast accuracy metrics.

All models are scored on exactly the same test points with the same MASE
scaling factor, which is computed once from the training data so that the
denominator does not change between models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_as_array(y_true) - _as_array(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((_as_array(y_true) - _as_array(y_pred)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean forecast error. Positive means the forecast is too high on average."""
    return float(np.mean(_as_array(y_pred) - _as_array(y_true)))


def seasonal_naive_scale(y_train, seasonality: int = config.DAILY_PERIOD) -> float:
    """In-sample mean absolute error of the seasonal naive forecast."""
    y_train = _as_array(y_train)

    if len(y_train) <= seasonality:
        raise ValueError("Training series is shorter than the seasonal period.")

    return float(np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])))


def mase(y_true, y_pred, y_train, seasonality: int = config.DAILY_PERIOD) -> float:
    """Mean absolute scaled error against the in-sample seasonal naive forecast."""
    scale = seasonal_naive_scale(y_train, seasonality=seasonality)

    if scale == 0:
        return float("nan")

    return mae(y_true, y_pred) / scale


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, reported only as a secondary diagnostic."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of a prediction interval."""
    y_true = _as_array(y_true)
    return float(np.mean((y_true >= _as_array(lower)) & (y_true <= _as_array(upper))))


def interval_width(lower, upper) -> float:
    """Average width of a prediction interval."""
    return float(np.mean(_as_array(upper) - _as_array(lower)))


def evaluate_forecast(name, y_true, y_pred, y_train, seasonality=config.DAILY_PERIOD) -> dict:
    """Score a single forecast and return the four required metrics."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(np.asarray(y_pred, dtype=float), index=y_true.index)

    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_all(forecasts: dict, y_true, y_train, seasonality=config.DAILY_PERIOD) -> pd.DataFrame:
    """Score a dictionary of forecasts and return a table sorted by MASE."""
    y_true = pd.Series(y_true).astype(float)

    rows = []
    for name, prediction in forecasts.items():
        prediction = pd.Series(prediction).reindex(y_true.index)
        valid = prediction.notna() & y_true.notna()

        rows.append(
            evaluate_forecast(
                name=name,
                y_true=y_true.loc[valid],
                y_pred=prediction.loc[valid],
                y_train=y_train,
                seasonality=seasonality,
            )
        )

    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)


def errors_by_hour(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Mean absolute error by hour of day, used in the error analysis."""
    residual = (pd.Series(y_pred).reindex(y_true.index) - y_true).abs()
    return residual.groupby(y_true.index.hour).mean()


def errors_by_horizon(y_true: pd.Series, y_pred: pd.Series, horizon=config.HORIZON) -> pd.Series:
    """Mean absolute error by position within the 24-hour forecast block."""
    residual = (pd.Series(y_pred).reindex(y_true.index) - y_true).abs()
    step = np.arange(len(y_true)) % horizon + 1
    return residual.groupby(step).mean()


def diebold_mariano(y_true, pred_a, pred_b, power: int = 1):
    """Diebold-Mariano test of equal predictive accuracy.

    Returns the test statistic and a two-sided p-value using a normal
    approximation with a Newey-West long-run variance estimate.
    """
    from scipy import stats

    y_true = _as_array(y_true)
    loss_a = np.abs(y_true - _as_array(pred_a)) ** power
    loss_b = np.abs(y_true - _as_array(pred_b)) ** power

    d = loss_a - loss_b
    n = len(d)
    d_bar = d.mean()

    # Newey-West variance with a rule-of-thumb bandwidth.
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    gamma0 = np.mean((d - d_bar) ** 2)
    variance = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        variance += 2 * (1 - k / (lag + 1)) * gamma_k

    variance = max(variance, 1e-12)
    statistic = d_bar / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))

    return float(statistic), float(p_value)
