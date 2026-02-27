"""
FCGMB Validation Test
=====================
Runs docking-based validation for all configured benchmarks.

For each benchmark the script:
  1. Auto-discovers benchmark names from fcgmb/configs/*.yaml (+ any local configs/).
  2. Loads the upper 75% activity compounds (validation set) from bundled or
     cached bioactivity data.
  3. Docks those compounds and reports normalized scores.
  4. Saves per-benchmark CSV and a combined summary CSV.

Usage
-----
  # Run all benchmarks:
  python validation_test.py

  # Run a subset:
  python validation_test.py --benchmarks AKT1 CHK1

  # Show lower-25% (initial) compounds table:
  python validation_test.py --show-initial

Extending
---------
  To add your own benchmark:
    1. Create fcgmb/configs/<MyBenchmark>.yaml  (see existing files for format)
    2. Add bioactivity data at fcgmb/bioactivity_data/<MyBenchmark>.csv
       (columns: molecule_chembl_id, canonical_smiles, pchembl_value)
    3. Place pre-built grids in fcgmb/grids/<PDB_ID>/  OR let the oracle
       auto-prepare them (requires autogrid4 + reduce2).
"""

import argparse
import os
import sys
from pathlib import Path

import polars as pl

from fcgmb.oracle import FCGMBOracle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(scores: list) -> dict:
    """Compute basic stats from a list of floats (NaN-safe)."""
    valid = [s for s in scores if s == s]  # NaN check
    return {
        "n_valid": len(valid),
        "min_score": min(valid) if valid else float("nan"),
        "max_score": max(valid) if valid else float("nan"),
        "mean_score": sum(valid) / len(valid) if valid else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def run_validation(
    benchmarks: list | None = None,
    include_user_configs: bool = False,
    show_initial: bool = False,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Run validation docking for the specified (or all) benchmarks.

    Parameters
    ----------
    benchmarks : list[str] | None
        Benchmark names to run. If None, all benchmarks discovered from
        configs are used.
    show_initial : bool
        If True, print and save the lower-25% (initial) compounds table.
    output_dir : Path | None
        Directory for output CSVs. Defaults to current working directory.

    Returns
    -------
    pl.DataFrame
        Summary table with one row per benchmark.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover benchmarks
    # By default use only the curated package benchmarks (fcgmb/configs/).
    # Pass include_user_configs=True to also pick up local configs/.
    all_available = FCGMBOracle.list_benchmarks()

    if benchmarks:
        to_run = benchmarks
    else:
        to_run = all_available

    if not to_run:
        print("[ERROR] No benchmark configs found. Check fcgmb/configs/ or local configs/.")
        sys.exit(1)

    invalid = [b for b in to_run if b not in all_available]
    if invalid:
        print(f"[ERROR] Unknown benchmark(s): {invalid}")
        print(f"Available: {all_available}")
        sys.exit(1)

    print(f"\nFCGMB Validation — {len(to_run)} benchmark(s): {to_run}")
    adgpu_exe = os.environ.get("ADGPU_EXE", "adgpu")

    summary_rows = []

    for bm_name in to_run:
        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {bm_name}")
        print(f"{'=' * 60}")

        try:
            oracle = FCGMBOracle(bm_name, budget=2000)
            oracle.set_backend_config(adgpu_executable=adgpu_exe)

            # ------------------------------------------------------------------
            # Optional: show the lower-25% (initial) compounds
            # ------------------------------------------------------------------
            if show_initial:
                initial_df = oracle.get_initial_compounds()
                if not initial_df.is_empty():
                    # Keep only the relevant columns for display
                    display_cols = [c for c in ["canonical_smiles", "pchembl_value"]
                                    if c in initial_df.columns]
                    print(f"\n--- Lower 25% compounds for {bm_name} ({len(initial_df)}) ---")
                    print(initial_df.select(display_cols))
                    out_initial = output_dir / f"initial_compounds_{bm_name}.csv"
                    initial_df.write_csv(out_initial)
                    print(f"  Saved to {out_initial}")

            # ------------------------------------------------------------------
            # Dock the upper 75% (validation set)
            # ------------------------------------------------------------------
            val_df = oracle.get_validation_compounds()
            if val_df.is_empty():
                print(f"[WARN] No validation compounds for {bm_name}, skipping.")
                continue

            smiles_list = val_df.get_column("canonical_smiles").to_list()
            print(f"Scoring {len(smiles_list)} validation compounds (upper 75%)...")

            results = oracle.score(smiles_list)
            scores = [results.get(smi, float("nan")) for smi in smiles_list]

            st = _stats(scores)
            n_success = int(oracle.results_df.get_column("valid_pose_found").sum()) \
                if not oracle.results_df.is_empty() else 0

            print(f"\n  Results for {bm_name}:")
            print(f"    Scored:          {len(scores)}")
            print(f"    Valid poses:     {n_success}")
            print(f"    Score range:     {st['min_score']:.3f} – {st['max_score']:.3f}")
            print(f"    Mean score:      {st['mean_score']:.3f}")

            # Save per-benchmark detailed results
            out_file = output_dir / f"validation_results_{bm_name}.csv"
            oracle.results_df.write_csv(out_file)
            print(f"    Saved results -> {out_file}")

            summary_rows.append({
                "benchmark":   bm_name,
                "n_scored":    len(scores),
                "n_valid_pose": n_success,
                "min_score":   st["min_score"],
                "max_score":   st["max_score"],
                "mean_score":  st["mean_score"],
            })

        except Exception as exc:
            import traceback
            print(f"[ERROR] {bm_name}: {exc}")
            traceback.print_exc()
            summary_rows.append({
                "benchmark": bm_name,
                "n_scored": 0, "n_valid_pose": 0,
                "min_score": float("nan"), "max_score": float("nan"),
                "mean_score": float("nan"),
            })

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    if summary_rows:
        summary_df = pl.DataFrame(summary_rows)
        print(summary_df)
        summary_out = output_dir / "validation_summary.csv"
        summary_df.write_csv(summary_out)
        print(f"\nSummary saved -> {summary_out}")
    else:
        summary_df = pl.DataFrame()
        print("No results to summarize.")

    return summary_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FCGMB validation: dock upper-75%% activity compounds for each benchmark."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also run user-added benchmarks from a local 'configs/' directory.",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        metavar="NAME",
        help="Benchmark name(s) to run (default: 6 bundled benchmarks). E.g. --benchmarks AKT1 CHK1",
    )
    parser.add_argument(
        "--show-initial",
        action="store_true",
        help="Print and save the lower-25%% (initial) compounds table per benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write output CSVs (default: current directory).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_validation(
        benchmarks=args.benchmarks,
        include_user_configs=args.all,
        show_initial=args.show_initial,
        output_dir=args.output_dir,
    )
