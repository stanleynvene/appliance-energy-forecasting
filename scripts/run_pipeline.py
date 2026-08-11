"""Run the full forecasting pipeline and write every output.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --foundation-backend bolt
    python scripts/run_pipeline.py --foundation-backend fallback --skip-single-block
"""

import argparse

import _bootstrap  # noqa: F401

from appliance_energy.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation-backend",
        default="chronos2",
        choices=["chronos2", "bolt", "fallback"],
        help="foundation model to use; 'fallback' forces the offline stand-in",
    )
    parser.add_argument("--skip-single-block", action="store_true",
                        help="skip the 336-step-ahead sensitivity run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        foundation_backend=args.foundation_backend,
        run_single_block=not args.skip_single_block,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
