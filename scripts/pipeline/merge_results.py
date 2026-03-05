#!/usr/bin/env python3
"""
Utility: Merge partial results from parallel SLURM jobs.

Usage:
    python merge_results.py --pattern "data/chembl_pdb_map_*.csv" --output data/chembl_pdb_map.csv
    python merge_results.py --pattern "data/chembl_docking_benchmark_*.csv" --output data/chembl_docking_benchmark.csv
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import merge_csv_files


def main():
    parser = argparse.ArgumentParser(description="Merge partial CSV results")
    parser.add_argument("--pattern", required=True, help="Glob pattern for input files")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    parser.add_argument(
        "--dedup-cols", nargs="*", help="Columns to use for deduplication"
    )
    args = parser.parse_args()

    merge_csv_files(args.pattern, args.output, args.dedup_cols)


if __name__ == "__main__":
    main()
