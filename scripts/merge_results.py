#!/usr/bin/env python3
"""
Utility: Merge partial results from parallel SLURM jobs.

Usage:
    python merge_results.py --pattern "data/chembl_pdb_map_*.csv" --output data/chembl_pdb_map.csv
    python merge_results.py --pattern "data/chembl_docking_benchmark_*.csv" --output data/chembl_docking_benchmark.csv
"""
import pandas as pd
import argparse
from pathlib import Path
import glob


def main():
    parser = argparse.ArgumentParser(description="Merge partial CSV results")
    parser.add_argument("--pattern", required=True, help="Glob pattern for input files")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    parser.add_argument("--dedup-cols", nargs="*", help="Columns to use for deduplication")
    args = parser.parse_args()

    # Find files
    files = sorted(glob.glob(args.pattern))
    print(f"Found {len(files)} files matching '{args.pattern}':")
    for f in files:
        print(f"  - {f}")

    if not files:
        print("No files found!")
        return

    # Load and concatenate
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {f}")
        except Exception as e:
            print(f"  [!] Error loading {f}: {e}")

    if not dfs:
        print("No data loaded!")
        return

    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after merge: {len(df_merged)}")

    # Deduplicate if columns specified
    if args.dedup_cols:
        before = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=args.dedup_cols)
        print(f"After deduplication on {args.dedup_cols}: {len(df_merged)} (removed {before - len(df_merged)})")

    # Save
    df_merged.to_csv(args.output, index=False)
    print(f"\nSaved merged results to {args.output}")


if __name__ == "__main__":
    main()
