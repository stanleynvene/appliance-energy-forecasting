"""Central configuration: paths, constants and forecasting design.

Everything that the rest of the package needs to agree on lives here, so that
the notebooks, the scripts and the tests all use identical settings.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"

RAW_FILENAME = "energydata_complete.csv"
RAW_CSV = RAW_DIR / RAW_FILENAME
HOURLY_CSV = PROCESSED_DIR / "appliance_hourly.csv"
FEATURE_CSV = PROCESSED_DIR / "appliance_features.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)

ALL_DIRS = [
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODEL_DIR,
    REPORT_FIGURE_DIR,
]


def ensure_dirs():
    """Create every output directory the pipeline writes to."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Forecasting design
# ------------------------------------------------------------------

RANDOM_STATE = 0

TARGET = "Appliances"

# The raw data are sampled every 10 minutes. We resample to hourly averages,
# as the README allows, so that SARIMAX estimation is tractable.
FREQ = "h"

DAILY_PERIOD = 24            # observations in one day at hourly frequency
WEEKLY_PERIOD = 24 * 7       # observations in one week at hourly frequency

# The forecasting task in the brief is "forecast the next 24 hours".
HORIZON = 24

# The recommended test period is the final 14 days.
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24  # 336 hourly observations

# The 14-day test period is evaluated as 14 consecutive 24-hour-ahead
# forecasts (a rolling-origin evaluation), so that every reported number
# refers to the 24-hour horizon actually specified by the brief.
N_ORIGINS = TEST_STEPS // HORIZON

# Weather variables. These are *realised* values in the test period, so any
# model that uses them is producing a conditional forecast.
WEATHER_COLS = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "Press_mm_hg",
]

# Indoor sensors.
INDOOR_TEMP_COLS = [f"T{i}" for i in range(1, 10)]
INDOOR_HUMIDITY_COLS = [f"RH_{i}" for i in range(1, 10)]

# Calendar features are genuinely known at the forecast origin.
CALENDAR_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

# Exogenous sets used by SARIMAX.
SARIMAX_EXOG_CALENDAR = CALENDAR_COLS
SARIMAX_EXOG_WEATHER = ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
SARIMAX_EXOG_FULL = SARIMAX_EXOG_CALENDAR + SARIMAX_EXOG_WEATHER

# SARIMAX orders. Chosen in notebook 04; the README's suggested starting point.
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)

# Lag and rolling-window definitions for the feature-based model.
# For a 24-hour-ahead forecast only lags of 24 hours or more are available at
# the forecast origin, which is why the operational feature set starts at 24.
LAGS_OPERATIONAL = [24, 25, 26, 48, 72, 168, 336]
LAGS_ONE_STEP = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 168]

# Column order required by the README for outputs/forecasts/all_forecasts.csv
REQUIRED_FORECAST_COLUMNS = [
    "actual",
    "mean",
    "naive",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
    "drift",
    "sarimax",
    "feature_model",
    "foundation_model",
]
