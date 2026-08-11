"""Figures for the exploratory analysis, the forecasts and the diagnostics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def save(fig, path, also_to_report: bool = True):
    """Save a figure to ``outputs/figures`` and mirror it into ``reports/figures``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")

    if also_to_report:
        config.REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.REPORT_FIGURE_DIR / path.name, bbox_inches="tight")

    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Exploratory figures
# ------------------------------------------------------------------

def plot_series_overview(hourly: pd.DataFrame, target: str = config.TARGET):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    hourly[target].plot(ax=axes[0], linewidth=0.7)
    axes[0].set_title("Hourly mean appliance energy use, full sample")
    axes[0].set_ylabel("Wh")
    axes[0].set_xlabel("")

    window = hourly[target].iloc[-24 * 21:]
    window.plot(ax=axes[1], linewidth=1.1, color="tab:red")
    axes[1].set_title("Final three weeks")
    axes[1].set_ylabel("Wh")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    return fig


def plot_seasonal_profiles(hourly: pd.DataFrame, target: str = config.TARGET):
    frame = hourly.copy()
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    hourly_profile = frame.groupby("hour")[target]
    axes[0].plot(hourly_profile.median().index, hourly_profile.median().values, marker="o")
    axes[0].fill_between(
        hourly_profile.quantile(0.25).index,
        hourly_profile.quantile(0.25).values,
        hourly_profile.quantile(0.75).values,
        alpha=0.25,
    )
    axes[0].set_title("Hour-of-day profile (median, IQR)")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Wh")

    dow_profile = frame.groupby("dayofweek")[target].median()
    axes[1].bar(dow_profile.index, dow_profile.values, color="tab:orange")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_title("Day-of-week profile (median)")
    axes[1].set_ylabel("Wh")

    axes[2].hist(hourly[target], bins=50, color="tab:green", alpha=0.85)
    axes[2].set_title("Distribution of hourly energy use")
    axes[2].set_xlabel("Wh")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 200):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], zero=False)
    axes[0].set_title("ACF of hourly appliance energy use")

    plot_pacf(series.dropna(), lags=min(lags, 60), ax=axes[1], zero=False, method="ywm")
    axes[1].set_title("PACF of hourly appliance energy use")

    fig.tight_layout()
    return fig


def plot_correlations(hourly: pd.DataFrame, target: str = config.TARGET, top_n: int = 15):
    correlations = (
        hourly.corr(numeric_only=True)[target].drop(target).sort_values(key=np.abs, ascending=False)
    )
    correlations = correlations.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:blue" if v > 0 else "tab:red" for v in correlations.values]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.set_title(f"Correlation with {target} (top {top_n} by magnitude)")
    ax.set_xlabel("Pearson correlation")

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Forecast figures
# ------------------------------------------------------------------

def plot_forecasts(train: pd.Series, test: pd.Series, forecast_df: pd.DataFrame,
                   columns=None, context_days: int = 7):
    """Test-period forecasts for the selected models, with recent history."""
    columns = [c for c in (columns or forecast_df.columns) if c != "actual"]

    fig, ax = plt.subplots(figsize=(14, 6))

    train.tail(context_days * 24).plot(ax=ax, label="Training data", linewidth=1.2, color="0.55")
    test.plot(ax=ax, label="Actual", linewidth=2.0, color="black")

    for column in columns:
        if column in forecast_df:
            forecast_df[column].plot(ax=ax, label=column, alpha=0.85, linewidth=1.2)

    ax.axvline(train.index[-1], linestyle=":", color="0.3")
    ax.set_title("Appliance energy forecasts over the 14-day test period (24-hour horizon)")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, loc="upper left")

    fig.tight_layout()
    return fig


def plot_forecast_zoom(test: pd.Series, forecast_df: pd.DataFrame, columns=None, days: int = 4):
    columns = [c for c in (columns or forecast_df.columns) if c != "actual"]
    window = slice(0, days * 24)

    fig, ax = plt.subplots(figsize=(13, 5))
    test.iloc[window].plot(ax=ax, label="Actual", color="black", linewidth=2.2)

    for column in columns:
        if column in forecast_df:
            forecast_df[column].iloc[window].plot(ax=ax, label=column, linewidth=1.3, alpha=0.9)

    for origin in range(days):
        ax.axvline(test.index[origin * 24], linestyle=":", color="0.6", linewidth=0.9)

    ax.set_title(f"First {days} days of the test period, dotted lines mark forecast origins")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend(ncol=3)

    fig.tight_layout()
    return fig


def plot_intervals(test: pd.Series, point: pd.Series, lower: pd.Series, upper: pd.Series,
                   label: str, days: int = 7):
    window = slice(0, days * 24)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(
        test.index[window], lower.iloc[window], upper.iloc[window],
        alpha=0.25, label=f"{label} 80% interval", color="tab:blue",
    )
    test.iloc[window].plot(ax=ax, color="black", linewidth=2.0, label="Actual")
    point.iloc[window].plot(ax=ax, color="tab:blue", linewidth=1.4, label=f"{label} median")

    ax.set_title(f"{label}: point forecast and 80% prediction interval")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend()

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def plot_error_diagnostics(test: pd.Series, forecast_df: pd.DataFrame, columns,
                           horizon: int = config.HORIZON):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for column in columns:
        if column not in forecast_df:
            continue
        error = forecast_df[column] - test

        by_hour = error.abs().groupby(test.index.hour).mean()
        axes[0, 0].plot(by_hour.index, by_hour.values, marker="o", label=column)

        step = np.arange(len(test)) % horizon + 1
        by_step = error.abs().groupby(step).mean()
        axes[0, 1].plot(by_step.index, by_step.values, marker="o", label=column)

        axes[1, 0].hist(error.dropna(), bins=40, alpha=0.5, label=column)

        axes[1, 1].scatter(test, forecast_df[column], s=8, alpha=0.5, label=column)

    axes[0, 0].set_title("Mean absolute error by hour of day")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("MAE (Wh)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Mean absolute error by step within the 24-hour block")
    axes[0, 1].set_xlabel("Steps ahead")
    axes[0, 1].set_ylabel("MAE (Wh)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Forecast error distribution")
    axes[1, 0].set_xlabel("Forecast minus actual (Wh)")
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].legend(fontsize=8)

    limits = [0, float(test.max()) * 1.05]
    axes[1, 1].plot(limits, limits, color="black", linewidth=1)
    axes[1, 1].set_title("Predicted against actual")
    axes[1, 1].set_xlabel("Actual (Wh)")
    axes[1, 1].set_ylabel("Predicted (Wh)")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_residual_acf(residuals: dict, lags: int = 48):
    from statsmodels.graphics.tsaplots import plot_acf

    n = len(residuals)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for ax, (name, series) in zip(axes[0], residuals.items()):
        plot_acf(pd.Series(series).dropna(), lags=lags, ax=ax, zero=False)
        ax.set_title(f"Residual ACF: {name}")
        ax.set_xlabel("Lag (hours)")

    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 20, title=None):
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top.values, color="tab:purple")
    ax.set_title(title or f"Feature importance (top {top_n})")
    ax.set_xlabel("Importance")

    fig.tight_layout()
    return fig


def plot_metric_comparison(results: pd.DataFrame, metric: str = "MASE"):
    frame = results.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:grey" if "naive" in m or m in {"mean", "drift"} else "tab:blue"
              for m in frame["model"]]
    ax.barh(frame["model"], frame[metric], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{metric} by model (lower is better; dashed line is MASE = 1)")
    ax.set_xlabel(metric)

    fig.tight_layout()
    return fig
