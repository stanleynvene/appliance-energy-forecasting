"""SARIMAX modelling of the hourly appliance energy series.

The model is estimated once on the initial training sample. At each subsequent
forecast origin the parameters are held fixed and the state-space filter is
re-run over the data observed so far (``results.apply(..., refit=False)``).
That updates the filtered state without re-estimating anything, which mirrors
how a deployed model would be run and keeps a 14-origin backtest
computationally reasonable.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config


def fit_sarimax(
    y_train: pd.Series,
    X_train=None,
    order=config.SARIMAX_ORDER,
    seasonal_order=config.SARIMAX_SEASONAL_ORDER,
    trend=None,
    maxiter: int = 100,
):
    """Estimate a SARIMAX model.

    ``trend`` defaults to ``None`` because the seasonal difference already
    removes a constant level; including a constant alongside ``D=1`` is not
    identified in the usual sense and slows convergence.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(disp=False, maxiter=maxiter)

    return fitted


def forecast_sarimax(fitted, horizon: int, index, X_test=None, alpha: float = 0.2):
    """Forecast ``horizon`` steps ahead and return the mean and an interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fitted.get_forecast(steps=horizon, exog=X_test)

        mean = pd.Series(np.asarray(result.predicted_mean, dtype=float), index=index, name="sarimax")
        conf = result.conf_int(alpha=alpha)

    lower = pd.Series(np.asarray(conf.iloc[:, 0], dtype=float), index=index, name="sarimax_lower")
    upper = pd.Series(np.asarray(conf.iloc[:, 1], dtype=float), index=index, name="sarimax_upper")

    return mean, lower, upper


def extend_sarimax(fitted, y_new: pd.Series, X_new=None):
    """Append newly observed data without re-estimating the parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.append(y_new, exog=X_new, refit=False)


def refilter_sarimax(fitted, y_history: pd.Series, X_history=None):
    """Re-run the filter over a longer history holding the parameters fixed.

    ``apply`` is used rather than repeated ``append`` calls because appending
    chains results objects together and the whole chain stays in memory for
    the length of a backtest.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.apply(y_history, exog=X_history, refit=False)


def rolling_origin_sarimax(
    y: pd.Series,
    exog: pd.DataFrame | None,
    n_origins: int,
    horizon: int,
    order=config.SARIMAX_ORDER,
    seasonal_order=config.SARIMAX_SEASONAL_ORDER,
    alpha: float = 0.2,
    name: str = "sarimax",
    long_horizon: int | None = None,
    keep_residuals: bool = False,
):
    """Run a rolling-origin backtest of a SARIMAX model.

    The final ``n_origins * horizon`` observations are forecast in blocks of
    ``horizon`` steps, re-filtering the fixed-parameter model at each origin.

    Optionally also returns a single ``long_horizon``-step forecast issued from
    the initial training sample, which is the design used by the demo pipeline.
    Only summary information is returned rather than the fitted results object,
    because state-space results are large and several models are estimated in
    one pipeline run.
    """
    test_steps = n_origins * horizon
    split = len(y) - test_steps

    y_train = y.iloc[:split]
    X_train = None if exog is None else exog.iloc[:split]

    fitted = fit_sarimax(
        y_train, X_train=X_train, order=order, seasonal_order=seasonal_order
    )

    diagnostics = summarise_fit(fitted, name=name)
    residuals = (
        pd.Series(np.asarray(fitted.resid, dtype=float), index=y_train.index)
        if keep_residuals
        else None
    )

    long_forecast = None
    if long_horizon:
        long_index = y.index[split:split + long_horizon]
        long_forecast, _, _ = forecast_sarimax(
            fitted,
            horizon=long_horizon,
            index=long_index,
            X_test=None if exog is None else exog.iloc[split:split + long_horizon],
            alpha=alpha,
        )
        long_forecast = long_forecast.rename(name)

    means, lowers, uppers = [], [], []

    for origin in range(n_origins):
        start = split + origin * horizon
        stop = start + horizon

        index = y.index[start:stop]
        X_future = None if exog is None else exog.iloc[start:stop]

        if origin == 0:
            current = fitted
        else:
            current = refilter_sarimax(
                fitted,
                y_history=y.iloc[:start],
                X_history=None if exog is None else exog.iloc[:start],
            )

        mean, lower, upper = forecast_sarimax(
            current, horizon=horizon, index=index, X_test=X_future, alpha=alpha
        )

        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

        if origin > 0:
            del current

    del fitted
    gc.collect()

    point = pd.concat(means).rename(name)
    lower = pd.concat(lowers).rename(f"{name}_lower")
    upper = pd.concat(uppers).rename(f"{name}_upper")

    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "diagnostics": diagnostics,
        "residuals": residuals,
        "long_forecast": long_forecast,
    }


def summarise_fit(fitted, name: str = "sarimax") -> dict:
    """A compact summary of a fitted model for the metrics folder."""
    return {
        "model": name,
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "loglikelihood": float(fitted.llf),
        "n_params": int(len(fitted.params)),
    }
