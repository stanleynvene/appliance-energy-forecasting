"""End-to-end forecasting pipeline.

Design
------
The brief asks for 24-hour-ahead forecasts and recommends the final 14 days as
the test period. Those two requirements are combined here as a rolling-origin
evaluation: the 336 hourly test observations are forecast as fourteen
consecutive blocks of 24 hours. Every model is re-issued at each origin using
only the data observed up to that point, so every number reported in
``outputs/metrics`` refers to the 24-hour horizon that the brief specifies.

A single 336-step-ahead run, matching the demo pipeline exactly, is also
produced as a sensitivity check and written to
``outputs/metrics/model_comparison_single_block.csv``.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from . import config, data, evaluation, features, plotting
from .models import benchmarks, feature_models, foundation, sarimax as sarimax_models

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def rolling_origins(y: pd.Series, n_origins: int = config.N_ORIGINS,
                    horizon: int = config.HORIZON) -> pd.DatetimeIndex:
    """Timestamps of the last observation available at each forecast origin."""
    split = len(y) - n_origins * horizon
    return pd.DatetimeIndex([y.index[split + k * horizon - 1] for k in range(n_origins)])


def rolling_benchmarks(y: pd.Series, n_origins: int = config.N_ORIGINS,
                       horizon: int = config.HORIZON) -> dict:
    """Re-issue every benchmark at each 24-hour origin and concatenate."""
    split = len(y) - n_origins * horizon
    collected: dict[str, list] = {}

    for origin in range(n_origins):
        start = split + origin * horizon
        stop = start + horizon

        history = y.iloc[:start]
        index = y.index[start:stop]

        for name, forecast in benchmarks.all_benchmarks(history, horizon, index).items():
            collected.setdefault(name, []).append(forecast)

    return {name: pd.concat(parts).rename(name) for name, parts in collected.items()}


def build_exog(hourly: pd.DataFrame) -> dict:
    """Exogenous blocks for SARIMAX, separated by what is knowable in advance."""
    calendar = features.time_feature_frame(hourly.index, columns=config.CALENDAR_COLS)

    weather_cols = features.available(hourly, config.SARIMAX_EXOG_WEATHER)
    weather = hourly[weather_cols].copy()

    # Standardise the weather block using training-period statistics only.
    split = len(hourly) - config.TEST_STEPS
    train_mean = weather.iloc[:split].mean()
    train_std = weather.iloc[:split].std().replace(0, 1.0)
    weather = (weather - train_mean) / train_std

    return {
        "calendar": calendar,
        "weather": weather,
        "full": pd.concat([calendar, weather], axis=1),
    }


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(
    foundation_backend: str = "chronos2",
    run_single_block: bool = True,
    verbose: bool = True,
):
    """Run every model, evaluate them and write all outputs to ``outputs/``."""
    started = time.time()
    config.ensure_dirs()

    np.random.seed(config.RANDOM_STATE)

    def log(message):
        if verbose:
            print(message, flush=True)

    # --------------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------------
    log("[1/9] Loading and preparing data")

    hourly = data.load_hourly()
    y = hourly[config.TARGET]

    train, test = data.train_test_split(y, test_steps=config.TEST_STEPS)
    horizon = config.HORIZON
    n_origins = config.N_ORIGINS

    log(f"      hourly observations: {len(hourly)}")
    log(f"      train: {train.index.min()} to {train.index.max()} ({len(train)} obs)")
    log(f"      test:  {test.index.min()} to {test.index.max()} ({len(test)} obs)")
    log(f"      evaluation: {n_origins} origins x {horizon}-hour horizon")

    origins = rolling_origins(y, n_origins, horizon)
    forecasts: dict[str, pd.Series] = {}
    intervals: dict[str, tuple] = {}
    notes: dict[str, str] = {}

    # --------------------------------------------------------------
    # 2. Benchmarks
    # --------------------------------------------------------------
    log("[2/9] Benchmark forecasts")

    forecasts.update(rolling_benchmarks(y, n_origins, horizon))
    for name in ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly",
                 "drift", "seasonal_mean_profile"]:
        notes[name] = "target only; known at forecast origin"

    # --------------------------------------------------------------
    # 3. SARIMAX
    # --------------------------------------------------------------
    log("[3/9] SARIMAX models (this is the slowest stage)")

    exog = build_exog(hourly)
    sarimax_summaries = []
    sarimax_residuals = None
    sarimax_single_block = None

    specifications = [
        ("sarimax_target_only", None, "target only; no covariates"),
        ("sarimax_calendar", exog["calendar"],
         "calendar covariates only; genuinely known in advance"),
        ("sarimax", exog["full"],
         "calendar plus realised weather; conditional forecast"),
    ]

    for name, exog_block, note in specifications:
        t0 = time.time()
        result = sarimax_models.rolling_origin_sarimax(
            y=y,
            exog=exog_block,
            n_origins=n_origins,
            horizon=horizon,
            name=name,
            keep_residuals=(name == "sarimax_target_only"),
            long_horizon=config.TEST_STEPS if (name == "sarimax" and run_single_block) else None,
        )

        forecasts[name] = result["point"]
        intervals[name] = (result["lower"], result["upper"])
        notes[name] = note
        sarimax_summaries.append(result["diagnostics"])

        if result["residuals"] is not None:
            sarimax_residuals = result["residuals"]
        if result["long_forecast"] is not None:
            sarimax_single_block = result["long_forecast"]

        log(f"      {name}: fitted and backtested in {time.time() - t0:.1f}s "
            f"(AIC {result['diagnostics']['aic']:.1f})")

    # --------------------------------------------------------------
    # 4. Feature-based models
    # --------------------------------------------------------------
    log(f"[4/9] Feature-based models (backend: {feature_models.backend_name()})")

    last_train_time = train.index[-1]
    first_test_time = test.index[0]

    direct_table = features.make_direct_table(
        hourly, horizon=horizon, include_sensors=True, include_future_weather=False
    )
    log(f"      operational design matrix: {direct_table.shape[0]} rows x "
        f"{len(features.direct_feature_columns(direct_table))} features")

    model_direct, pred_direct, direct_cols, direct_train_rows = feature_models.run_direct_model(
        direct_table, origins, last_train_time, first_test_time, name="feature_model"
    )
    forecasts["feature_model"] = pred_direct.reindex(test.index)
    notes["feature_model"] = (
        "gradient boosting; all predictors observable at the forecast origin"
    )

    conditional_table = features.make_direct_table(
        hourly, horizon=horizon, include_sensors=True, include_future_weather=True
    )
    model_cond, pred_cond, cond_cols, _ = feature_models.run_direct_model(
        conditional_table, origins, last_train_time, first_test_time,
        name="feature_model_conditional",
    )
    forecasts["feature_model_conditional"] = pred_cond.reindex(test.index)
    notes["feature_model_conditional"] = (
        "as above plus realised future weather; conditional forecast"
    )

    one_step_table = features.make_one_step_table(hourly)
    model_one_step, pred_one_step, one_step_cols = feature_models.run_one_step_model(
        one_step_table, test.index
    )
    forecasts["feature_model_one_step"] = pred_one_step.reindex(test.index)
    notes["feature_model_one_step"] = (
        "one-step design from the demo pipeline; uses lag_1, so NOT a valid "
        "24-hour-ahead forecast. Reported as a leakage diagnostic only."
    )

    importance_direct = feature_models.feature_importance(model_direct, direct_cols)

    # --------------------------------------------------------------
    # 5. Foundation model
    # --------------------------------------------------------------
    log(f"[5/9] Foundation model (requested backend: {foundation_backend})")

    point, lower, upper, foundation_meta = foundation.rolling_origin_foundation(
        y=y, n_origins=n_origins, horizon=horizon,
        backend=foundation_backend, name="foundation_model",
    )
    forecasts["foundation_model"] = point
    intervals["foundation_model"] = (lower, upper)

    if foundation_meta.get("used_fallback"):
        notes["foundation_model"] = (
            "OFFLINE FALLBACK (seasonal quantile model), not Chronos: "
            f"{foundation_meta.get('fallback_reason', 'unknown reason')}"
        )
        log("      Chronos unavailable; documented offline fallback used.")
        log(f"      reason: {foundation_meta.get('fallback_reason')}")
    else:
        notes["foundation_model"] = (
            f"zero-shot {foundation_meta.get('backend')}; target history only"
        )
        log(f"      backend actually used: {foundation_meta.get('backend')}")

    with open(config.METRICS_DIR / "foundation_model_backend.json", "w") as handle:
        json.dump(
            {
                "requested_backend": foundation_meta.get("requested_backend"),
                "backend_used": foundation_meta.get("backend"),
                "used_fallback": bool(foundation_meta.get("used_fallback")),
                "fallback_reason": foundation_meta.get("fallback_reason"),
            },
            handle,
            indent=2,
        )

    # --------------------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------------------
    log("[6/9] Evaluating forecasts")

    results = evaluation.evaluate_all(
        forecasts, y_true=test, y_train=train, seasonality=config.DAILY_PERIOD
    )
    results["notes"] = results["model"].map(notes)

    best_benchmark = (
        results.loc[results["model"].isin(
            ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly",
             "drift", "seasonal_mean_profile"]
        )]
        .sort_values("MASE")
        .iloc[0]
    )
    results["MASE_vs_best_benchmark"] = results["MASE"] / best_benchmark["MASE"]

    log("\n" + results.drop(columns=["notes"]).round(3).to_string(index=False) + "\n")
    log(f"      strongest benchmark: {best_benchmark['model']} "
        f"(MASE {best_benchmark['MASE']:.3f})")

    # Interval performance where a predictive distribution is available.
    interval_rows = []
    for name, (lower, upper) in intervals.items():
        interval_rows.append(
            {
                "model": name,
                "nominal": 0.80,
                "coverage": evaluation.coverage(test, lower.reindex(test.index),
                                                upper.reindex(test.index)),
                "average_width": evaluation.interval_width(
                    lower.reindex(test.index), upper.reindex(test.index)
                ),
            }
        )
    interval_df = pd.DataFrame(interval_rows).sort_values("model").reset_index(drop=True)

    # Error breakdowns.
    breakdown_models = [
        "seasonal_naive_daily", "seasonal_naive_weekly", "sarimax",
        "feature_model", "foundation_model",
    ]
    breakdown_models = [m for m in breakdown_models if m in forecasts]

    by_hour = pd.DataFrame(
        {m: evaluation.errors_by_hour(test, forecasts[m]) for m in breakdown_models}
    )
    by_horizon = pd.DataFrame(
        {m: evaluation.errors_by_horizon(test, forecasts[m], horizon) for m in breakdown_models}
    )

    # Diebold-Mariano tests against the strongest benchmark.
    dm_rows = []
    reference = forecasts[best_benchmark["model"]].reindex(test.index)
    for name in breakdown_models + ["feature_model_conditional", "sarimax_calendar"]:
        if name not in forecasts or name == best_benchmark["model"]:
            continue
        statistic, p_value = evaluation.diebold_mariano(
            test, reference, forecasts[name].reindex(test.index)
        )
        dm_rows.append(
            {
                "model": name,
                "reference": best_benchmark["model"],
                "DM_statistic": statistic,
                "p_value": p_value,
                "better_than_reference": bool(statistic > 0 and p_value < 0.05),
            }
        )
    dm_df = pd.DataFrame(dm_rows)

    # --------------------------------------------------------------
    # 7. Single-block sensitivity check (demo-pipeline design)
    # --------------------------------------------------------------
    single_block_results = None

    if run_single_block:
        log("[7/9] Single 336-step-ahead run (demo-pipeline design)")

        single_horizon = config.TEST_STEPS
        single = benchmarks.all_benchmarks(train, single_horizon, test.index)

        # The initial SARIMAX fit was estimated on exactly the training sample,
        # so its 336-step forecast is the demo pipeline's design.
        if sarimax_single_block is not None:
            single["sarimax"] = sarimax_single_block

        point_fm, _, _, _ = foundation.rolling_origin_foundation(
            y=y, n_origins=1, horizon=single_horizon,
            backend=foundation_backend, name="foundation_model",
        )
        single["foundation_model"] = point_fm

        single["feature_model"] = forecasts["feature_model"]

        single_block_results = evaluation.evaluate_all(single, test, train)
        log("\n" + single_block_results.round(3).to_string(index=False) + "\n")
    else:
        log("[7/9] Single-block run skipped")

    # --------------------------------------------------------------
    # 8. Figures
    # --------------------------------------------------------------
    log("[8/9] Writing figures")

    forecast_df = pd.DataFrame({"actual": test})
    for name, prediction in forecasts.items():
        forecast_df[name] = prediction.reindex(test.index)

    plotting.save(plotting.plot_series_overview(hourly), config.FIGURE_DIR / "series_overview.png")
    plotting.save(plotting.plot_seasonal_profiles(hourly), config.FIGURE_DIR / "seasonal_profiles.png")
    plotting.save(plotting.plot_acf_pacf(train), config.FIGURE_DIR / "acf_pacf.png")
    plotting.save(plotting.plot_correlations(hourly), config.FIGURE_DIR / "correlations.png")

    headline = ["seasonal_naive_daily", "seasonal_naive_weekly", "sarimax",
                "feature_model", "foundation_model"]

    plotting.save(
        plotting.plot_forecasts(train, test, forecast_df, columns=headline),
        config.FIGURE_DIR / "forecast_comparison.png",
    )
    plotting.save(
        plotting.plot_forecast_zoom(test, forecast_df, columns=headline, days=4),
        config.FIGURE_DIR / "forecast_zoom.png",
    )
    plotting.save(
        plotting.plot_error_diagnostics(test, forecast_df, headline),
        config.FIGURE_DIR / "error_diagnostics.png",
    )

    residuals = {
        "SARIMAX (in-sample)": sarimax_residuals,
        "feature_model (test)": (forecast_df["feature_model"] - test).dropna(),
        "foundation_model (test)": (forecast_df["foundation_model"] - test).dropna(),
    }
    plotting.save(plotting.plot_residual_acf(residuals), config.FIGURE_DIR / "residual_acf.png")

    if not importance_direct.empty:
        plotting.save(
            plotting.plot_feature_importance(
                importance_direct, title="Feature importance: operational 24-hour model"
            ),
            config.FIGURE_DIR / "feature_importance.png",
        )

    plotting.save(
        plotting.plot_metric_comparison(results, metric="MASE"),
        config.FIGURE_DIR / "metric_comparison.png",
    )

    for name in intervals:
        lower, upper = intervals[name]
        plotting.save(
            plotting.plot_intervals(
                test, forecasts[name].reindex(test.index),
                lower.reindex(test.index), upper.reindex(test.index), label=name,
            ),
            config.FIGURE_DIR / f"intervals_{name}.png",
        )

    # --------------------------------------------------------------
    # 9. Save tables
    # --------------------------------------------------------------
    log("[9/9] Writing forecasts and metrics")

    required = forecast_df[[c for c in config.REQUIRED_FORECAST_COLUMNS if c in forecast_df]]
    required.to_csv(config.FORECAST_DIR / "all_forecasts.csv")
    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts_extended.csv")

    interval_frame = pd.DataFrame({"actual": test})
    for name, (lower, upper) in intervals.items():
        interval_frame[f"{name}_lower"] = lower.reindex(test.index)
        interval_frame[f"{name}_median"] = forecasts[name].reindex(test.index)
        interval_frame[f"{name}_upper"] = upper.reindex(test.index)
    interval_frame.to_csv(config.FORECAST_DIR / "prediction_intervals.csv")

    results[["model", "MAE", "RMSE", "MASE", "Bias"]].to_csv(
        config.METRICS_DIR / "model_comparison.csv", index=False
    )
    results.to_csv(config.METRICS_DIR / "model_comparison_annotated.csv", index=False)
    interval_df.to_csv(config.METRICS_DIR / "interval_coverage.csv", index=False)
    by_hour.to_csv(config.METRICS_DIR / "mae_by_hour.csv")
    by_horizon.to_csv(config.METRICS_DIR / "mae_by_horizon.csv")
    dm_df.to_csv(config.METRICS_DIR / "diebold_mariano.csv", index=False)
    pd.DataFrame(sarimax_summaries).to_csv(
        config.METRICS_DIR / "sarimax_fit_summary.csv", index=False
    )

    if not importance_direct.empty:
        importance_direct.rename("importance").to_csv(
            config.METRICS_DIR / "feature_importance.csv"
        )

    if single_block_results is not None:
        single_block_results.to_csv(
            config.METRICS_DIR / "model_comparison_single_block.csv", index=False
        )

    run_info = {
        "run_seconds": round(time.time() - started, 1),
        "n_hourly_observations": int(len(hourly)),
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "horizon_hours": horizon,
        "n_origins": n_origins,
        "mase_scale_seasonal_naive_24h": evaluation.seasonal_naive_scale(train, 24),
        "feature_model_backend": feature_models.backend_name(),
        "foundation_backend_used": foundation_meta.get("backend"),
        "foundation_used_fallback": bool(foundation_meta.get("used_fallback")),
        "best_benchmark": best_benchmark["model"],
        "best_overall": results.iloc[0]["model"],
        "random_state": config.RANDOM_STATE,
    }
    with open(config.METRICS_DIR / "run_info.json", "w") as handle:
        json.dump(run_info, handle, indent=2, default=str)

    log(f"\nDone in {run_info['run_seconds']}s. Outputs written to {config.OUTPUT_DIR}")

    return {
        "results": results,
        "forecasts": forecast_df,
        "intervals": interval_df,
        "by_hour": by_hour,
        "by_horizon": by_horizon,
        "diebold_mariano": dm_df,
        "single_block": single_block_results,
        "feature_importance": importance_direct,
        "run_info": run_info,
    }
