#!/usr/bin/env python3
import argparse
from pathlib import Path
from fcgmb.variance import run_variance_tests, analyze_variance_results


def main():
    parser = argparse.ArgumentParser(
        description="Run and analyze variance tests for benchmarks"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing benchmark configs",
    )
    parser.add_argument(
        "--run-dir", type=str, default="variance_runs", help="Base directory for runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="variance_analysis",
        help="Directory for analysis output",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of docking iterations"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running docking, only run analysis",
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    if not args.skip_run:
        run_variance_tests(
            config_dir=config_dir, run_base_dir=run_dir, n_iterations=args.iterations
        )

    analyze_variance_results(
        run_base_dir=run_dir, config_dir=config_dir, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
