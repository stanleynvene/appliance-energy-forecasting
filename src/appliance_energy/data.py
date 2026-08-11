"""Loading, cleaning and resampling of the Appliances Energy Prediction data."""

from __future__ import annotations

import pandas as pd

from . import config


DROP_COLS = ["rv1", "rv2"]  # two random variables included in the original file


def download_raw(url: str = config.UCI_URL, destination=None, force: bool = False):
    """Download the raw 10-minute CSV from the UCI repository.

    The file is only downloaded if it is not already present, so a fresh clone
    with the data already in ``data/raw`` will not hit the network.
    """
    destination = config.RAW_CSV if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Raw file already present: {destination}")
        return destination

    print(f"Downloading {url}")
    frame = pd.read_csv(url)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame)} rows to {destination}")

    return destination


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw 10-minute data with a proper DatetimeIndex."""
    path = config.RAW_CSV if path is None else path

    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()

    frame = frame[~frame.index.duplicated(keep="first")]

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.drop(columns=[c for c in DROP_COLS if c in frame.columns])
    frame = frame.dropna(subset=[config.TARGET])

    return frame


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-minute observations to hourly means.

    Averaging rather than summing keeps the target on its original Wh scale
    and makes the hourly series directly comparable with the raw series.
    Any hour that is missing entirely is filled by time interpolation so that
    the index is strictly regular, which SARIMAX and Chronos both require.
    """
    hourly = frame.resample(config.FREQ).mean()

    n_missing = int(hourly[config.TARGET].isna().sum())
    if n_missing:
        print(f"Interpolating {n_missing} missing hourly observations.")

    hourly = hourly.interpolate(method="time").bfill().ffill()

    # A complete, strictly regular hourly index.
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq=config.FREQ)
    hourly = hourly.reindex(full_index).interpolate(method="time").bfill().ffill()
    hourly.index.name = "date"

    return hourly


def load_hourly(path=None, force_rebuild: bool = False) -> pd.DataFrame:
    """Return the cleaned hourly dataset, building and caching it if needed."""
    path = config.HOURLY_CSV if path is None else path

    if path.exists() and not force_rebuild:
        hourly = pd.read_csv(path, index_col=0, parse_dates=True)
        hourly.index.name = "date"
        return hourly

    hourly = to_hourly(load_raw())
    path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(path)

    return hourly


def train_test_split(frame, test_steps: int = config.TEST_STEPS):
    """Split chronologically: everything before the final ``test_steps`` is training."""
    if len(frame) <= test_steps:
        raise ValueError(
            f"Series of length {len(frame)} is too short for a {test_steps}-step test set."
        )

    return frame.iloc[:-test_steps], frame.iloc[-test_steps:]


def describe_series(series: pd.Series) -> pd.Series:
    """A small summary used in the exploratory notebook and the report."""
    return pd.Series(
        {
            "n": len(series),
            "start": series.index.min(),
            "end": series.index.max(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q25": series.quantile(0.25),
            "median": series.median(),
            "q75": series.quantile(0.75),
            "max": series.max(),
            "skew": series.skew(),
        }
    )
