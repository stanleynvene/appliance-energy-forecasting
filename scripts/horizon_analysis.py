"""Separate the forecast horizon from the time of day.

The main backtest issues one forecast per day, always at 18:00. Because every
origin sits at the same clock time, step ``h`` of the horizon always lands on
hour ``(18 + h) mod 24``: the mean error by horizon is then an exact rotation
of the mean error by hour of day, and tells us nothing about how accuracy
decays with lead time.

This script breaks that confound by re-issuing the operational feature model
from *every* hour in the test period rather than once a day. Each horizon step
is then averaged over all 24 times of day, so the resulting curve isolates the
effect of lead time.

Usage:
    python scripts/horizon_analysis.py
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from appliance_energy import config, data, evaluation, features, plotting
from appliance_energy.models import feature_models


def main():
    config.ensure_dirs()

    hourly = data.load_hourly()
    y = hourly[config.TARGET]
    train, test = data.train_test_split(y, test_steps=config.TEST_STEPS)

    table = features.make_direct_table(hourly, horizon=config.HORIZON)
    feature_cols = features.direct_feature_columns(table)

    train_rows = table.loc[table["target_time"] <= train.index[-1]]
    test_rows = table.loc[table["target_time"] >= test.index[0]].copy()

    print(f"Training rows: {len(train_rows)}")
    print(f"Evaluation rows: {len(test_rows)} "
          f"({test_rows['origin'].nunique()} distinct origins)")

    model = feature_models.fit_feature_model(train_rows[feature_cols], train_rows["y"])
    test_rows["prediction"] = model.predict(test_rows[feature_cols])
    test_rows["absolute_error"] = (test_rows["prediction"] - test_rows["y"]).abs()

    scale = evaluation.seasonal_naive_scale(train, config.DAILY_PERIOD)

    by_horizon = test_rows.groupby("h")["absolute_error"].agg(["mean", "median", "count"])
    by_horizon["MASE"] = by_horizon["mean"] / scale
    by_horizon = by_horizon.rename(columns={"mean": "MAE", "median": "median_AE"})

    # The daily-origin design for comparison: only origins at 18:00.
    daily_only = test_rows.loc[test_rows["origin"].dt.hour == train.index[-1].hour]
    daily_curve = daily_only.groupby("h")["absolute_error"].mean()
    by_horizon["MAE_daily_origins_only"] = daily_curve

    print("\nMean absolute error by horizon step:")
    print(by_horizon.round(2).to_string())

    slope = np.polyfit(by_horizon.index, by_horizon["MAE"], 1)[0]
    print(f"\nLinear trend across the horizon: {slope:+.3f} Wh per hour of lead time")
    print(f"MAE at h=1: {by_horizon['MAE'].iloc[0]:.2f} Wh")
    print(f"MAE at h=24: {by_horizon['MAE'].iloc[-1]:.2f} Wh")

    output = config.METRICS_DIR / "horizon_analysis.csv"
    by_horizon.to_csv(output)
    print(f"\nSaved {output}")

    fig, ax = plotting.plt.subplots(figsize=(9, 5))
    ax.plot(by_horizon.index, by_horizon["MAE"], marker="o",
            label="All hourly origins (horizon effect isolated)")
    ax.plot(by_horizon.index, by_horizon["MAE_daily_origins_only"], marker="s",
            linestyle="--", alpha=0.8,
            label="Daily origins only (confounded with hour of day)")
    ax.set_title("Feature model: mean absolute error against forecast lead time")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("MAE (Wh)")
    ax.legend()
    fig.tight_layout()

    plotting.save(fig, config.FIGURE_DIR / "horizon_analysis.png")
    print(f"Saved {config.FIGURE_DIR / 'horizon_analysis.png'}")


if __name__ == "__main__":
    main()
