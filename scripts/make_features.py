"""Build the processed hourly dataset and the supervised feature tables.

Usage:
    python scripts/make_features.py
"""

import _bootstrap  # noqa: F401

from appliance_energy import config, data, features


def main():
    config.ensure_dirs()

    hourly = data.load_hourly(force_rebuild=True)
    print(f"Hourly dataset: {hourly.shape[0]} rows, {hourly.shape[1]} columns")
    print(f"Saved to {config.HOURLY_CSV}")

    one_step = features.build_feature_dataset(hourly, save_path=config.FEATURE_CSV)
    print(f"One-step feature table: {one_step.shape}")

    direct = features.make_direct_table(hourly)
    direct_path = config.PROCESSED_DIR / "direct_design_matrix.csv"
    direct.to_csv(direct_path, index=False)
    print(f"Direct (origin, horizon) design matrix: {direct.shape}")
    print(f"Saved to {direct_path}")


if __name__ == "__main__":
    main()
