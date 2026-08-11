# Appliance Energy Forecasting

Short-term forecasting of household appliance energy use, comparing simple benchmarks, a
SARIMAX model, a feature-based gradient boosting model, and a zero-shot time-series
foundation model (Chronos-2).

The full write-up is in [`reports/report.md`](reports/report.md).

## Headline result

Forecasting 24 hours ahead over the final 14 days of the dataset, evaluated as fourteen
consecutive daily forecasts:

| Model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| Chronos-2 (zero-shot) | 32.82 | 67.09 | **0.614** | -18.65 |
| SARIMA(1,0,1)(1,1,1)[24], target only | 36.45 | 64.59 | 0.682 | -5.68 |
| XGBoost, operational features | 36.70 | 64.30 | 0.687 | -5.48 |
| SARIMAX + calendar | 37.51 | 64.54 | 0.702 | -5.71 |
| **Hour-of-week mean profile (best benchmark)** | 38.06 | **63.74** | 0.712 | -3.39 |
| SARIMAX + calendar + weather | 38.08 | 65.34 | 0.713 | -6.13 |
| XGBoost + realised future weather | 40.86 | 64.72 | 0.765 | 5.77 |
| Weekly seasonal naive | 43.46 | 81.41 | 0.813 | -13.16 |
| Daily seasonal naive | 48.31 | 85.57 | 0.904 | 1.75 |
| Mean | 50.26 | 74.94 | 0.941 | -3.29 |
| Naive | 85.55 | 110.39 | 1.601 | 50.98 |
| Drift | 85.80 | 110.68 | 1.606 | 51.37 |

Only Chronos-2 beats the strongest benchmark by a statistically significant margin
(Diebold-Mariano p = 0.018). Note that it has the best MAE but the *worst* RMSE of the
serious models, and that the best benchmark wins on RMSE outright.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

The raw CSV is committed to `data/raw/`, so a fresh clone runs without downloading
anything. The whole pipeline takes about six minutes on a laptop, most of it SARIMAX
estimation. (`scripts/download_data.py` will fetch the file from UCI if it is ever missing.)

To skip the 2.5 GB PyTorch install and use the documented offline stand-in for the
foundation model:

```bash
pip install -r requirements-minimal.txt
python scripts/run_pipeline.py --foundation-backend fallback
```

Check `outputs/metrics/foundation_model_backend.json` afterwards to confirm which backend
actually ran. It is never assumed.

## Repository layout

```
├── data/
│   ├── raw/                    energydata_complete.csv, 10-minute resolution
│   └── processed/              hourly dataset and feature tables (generated)
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
├── src/appliance_energy/
│   ├── config.py               paths, constants, forecasting design
│   ├── data.py                 loading, cleaning, resampling, splitting
│   ├── features.py             time, lag, rolling and covariate features
│   ├── evaluation.py           MAE, RMSE, MASE, bias, coverage, Diebold-Mariano
│   ├── plotting.py             all figures
│   ├── pipeline.py             end-to-end orchestration
│   └── models/
│       ├── benchmarks.py
│       ├── sarimax.py
│       ├── feature_models.py
│       └── foundation.py       Chronos-2 / Chronos-Bolt, with offline fallback
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py         main entry point
│   ├── evaluate_models.py      re-score saved forecasts without refitting
│   └── horizon_analysis.py     separates lead time from time of day
├── outputs/
│   ├── forecasts/              all_forecasts.csv and variants
│   ├── metrics/                model_comparison.csv and diagnostics
│   └── figures/
├── reports/report.md           the write-up
└── tests/                      31 tests
```

## Forecasting design

The brief asks for a 24-hour horizon and recommends the final 14 days as the test period.
These are combined as a **rolling-origin evaluation**: the 336 test observations are
forecast as fourteen consecutive 24-hour blocks, with every model re-issued at each origin
using only data observed up to that point. Every headline number therefore describes
accuracy at the 24-hour horizon.

A single 336-step-ahead run, matching the demo pipeline, is produced as a sensitivity check
and written to `outputs/metrics/model_comparison_single_block.csv`. The substantive
conclusions are unchanged; only the naive and drift benchmarks differ materially.

## Data leakage

Three distinctions are enforced throughout and documented in the results tables.

**Future values of the target.** All lag and rolling features are shifted so that no
feature uses the value being predicted. Verified in `tests/test_features.py`.

**Information unavailable at the forecast origin.** This is the subtler one. A feature
shifted by one hour relative to the *target* is not necessarily known at an origin 24 hours
earlier. The operational feature model uses only quantities measurable at the origin, plus
calendar features of the target time. A one-step model using `lag_1` is also reported, and
scores better (MASE 0.602), but is labelled as a leakage diagnostic and excluded from every
ranking.

**Covariates that would not be forecastable.** Models using realised test-period weather
are labelled as conditional forecasts and reported separately. They performed worse than
the operational versions, so nothing depends on the distinction, but it is maintained
anyway.

Scaling statistics are computed on the training sample only, and no model was selected on
test-set performance: the SARIMAX order follows the brief's suggestion and the ACF/PACF
diagnostics, and the gradient boosting hyperparameters were fixed a priori.

## Tests

```bash
python -m pytest
```

31 tests covering feature leakage, metric correctness (including MASE = 0 for a perfect
forecast), benchmark behaviour, forecast lengths, and the regularity of the processed
dataset.

## Reproducibility notes

`random_state = 0` throughout. The pipeline writes `outputs/metrics/run_info.json`
recording the split dates, the MASE scaling factor, the backends used and the runtime.

Two caveats worth stating. Gradient boosting is not bit-identical across platforms and
thread counts, so XGBoost figures may vary by a few percent between machines; the results
here were produced on Windows 11 with Python 3.13, pandas 3.0.5, scikit-learn 1.9 and
xgboost 3.4. And the original demo pipeline calls
`mean_squared_error(..., squared=False)`, which was removed in scikit-learn 1.6; this
project computes RMSE directly and is unaffected.
