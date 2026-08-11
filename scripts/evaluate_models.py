"""Re-score saved forecasts without refitting anything.

Reads ``outputs/forecasts/all_forecasts_extended.csv`` and rewrites the
metrics tables. Useful for checking the numbers quoted in the report.

Usage:
    python scripts/evaluate_models.py
"""

import pandas as pd

import _bootstrap  # noqa: F401

from appliance_energy import config, data, evaluation


def main():
    config.ensure_dirs()

    path = config.FORECAST_DIR / "all_forecasts_extended.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/run_pipeline.py first.")

    forecasts = pd.read_csv(path, index_col=0, parse_dates=True)

    hourly = data.load_hourly()
    train, test = data.train_test_split(hourly[config.TARGET])

    actual = forecasts["actual"]
    columns = [c for c in forecasts.columns if c != "actual"]

    results = evaluation.evaluate_all(
        {c: forecasts[c] for c in columns}, y_true=actual, y_train=train
    )

    print(results.round(3).to_string(index=False))
    results.to_csv(config.METRICS_DIR / "model_comparison_recomputed.csv", index=False)
    print(f"\nWritten to {config.METRICS_DIR / 'model_comparison_recomputed.csv'}")


if __name__ == "__main__":
    main()
