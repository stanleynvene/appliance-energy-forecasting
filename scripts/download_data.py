"""Download the raw Appliances Energy Prediction CSV into ``data/raw``.

Usage:
    python scripts/download_data.py [--force]
"""

import argparse

import _bootstrap  # noqa: F401

from appliance_energy import config, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    config.ensure_dirs()
    path = data.download_raw(force=args.force)

    frame = data.load_raw(path)
    print(f"Raw 10-minute observations: {len(frame)}")
    print(f"Period: {frame.index.min()} to {frame.index.max()}")


if __name__ == "__main__":
    main()
