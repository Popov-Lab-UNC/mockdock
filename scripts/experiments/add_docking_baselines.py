#!/usr/bin/env python3
"""add_docking_baselines.py – Phase 2 one-time helper.

Reads the five variance runs (variance_runs/run_1 … run_5) and computes a
mean docking score for each compound across the repeats.  The result is
written back into the six canonical bioactivity CSVs
(fcgmb/bioactivity_data/<benchmark>.csv) as two new columns:

  mean_docking_score  – raw mean docking score in kcal/mol (lower = better)
  score               – normalised score using each benchmark's low_score /
                        high_score calibration points. Formula:
                        (low_score - mean_docking_score) / (low_score - high_score).
                        Can be negative or exceed 1.0 when mean_docking_score
                        falls outside the calibration range.

Run once from the benchmark root::

    python add_docking_baselines.py

The script is idempotent: if the columns already exist they are overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_ROOT = Path(__file__).resolve().parent
VARIANCE_RUNS_DIR = BENCHMARK_ROOT / "variance_runs"
BIOACTIVITY_DIR = BENCHMARK_ROOT / "fcgmb" / "bioactivity_data"
CONFIGS_DIR = BENCHMARK_ROOT / "fcgmb" / "configs"

N_RUNS = 5  # variance_runs/run_1 … run_5


# Map each canonical benchmark name to its (target_id, pdb_id, doc_id, assay_id)
# so we can reconstruct the path to each variant run results CSV.
def _load_benchmark_config(benchmark_name: str) -> dict:
    config_path = CONFIGS_DIR / f"{benchmark_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _find_results_csv(
    run_dir: Path, target_id: str, pdb_id: str, doc_id: str
) -> Path | None:
    """Locate the per-compound results CSV inside a single variance run folder.

    Expected layout::
        run_N/<target_id>_<pdb_id>/<doc_id>/<target_id>_<pdb_id>_<doc_id>_*_results.csv
    """
    subdir = run_dir / f"{target_id}_{pdb_id}"
    if not subdir.exists():
        return None

    doc_subdir = subdir / doc_id
    if not doc_subdir.exists():
        # Some runs may store the results directly under the target subdir
        candidates = list(subdir.glob(f"*_results.csv"))
        return candidates[0] if candidates else None

    candidates = list(doc_subdir.glob("*_results.csv"))
    return candidates[0] if candidates else None


def process_benchmark(benchmark_name: str) -> None:
    csv_path = BIOACTIVITY_DIR / f"{benchmark_name}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {benchmark_name}: bioactivity CSV not found at {csv_path}")
        return

    cfg = _load_benchmark_config(benchmark_name)
    target_id = cfg["target_id"]
    pdb_id = cfg["pdb_id"]
    doc_id = cfg["doc_id"]
    low_score: float | None = cfg.get("low_score")
    high_score: float | None = cfg.get("high_score")

    if low_score is None or high_score is None:
        print(
            f"  [WARN] {benchmark_name}: low_score/high_score not in config; "
            "will add mean_docking_score but not normalised score."
        )

    # ── Collect per-compound docking scores from all variance runs ──────────
    all_run_frames: list[pl.DataFrame] = []
    for run_idx in range(1, N_RUNS + 1):
        run_dir = VARIANCE_RUNS_DIR / f"run_{run_idx}"
        results_csv = _find_results_csv(run_dir, target_id, pdb_id, doc_id)
        if results_csv is None:
            print(
                f"  [WARN] {benchmark_name}: no results CSV in run_{run_idx}, skipping."
            )
            continue

        run_df = pl.read_csv(results_csv)
        # Keep only chembl ID and docking score; tag the run number
        if (
            "molecule_chembl_id" not in run_df.columns
            or "docking_score" not in run_df.columns
        ):
            print(
                f"  [WARN] {benchmark_name}: run_{run_idx} CSV missing expected columns, skipping."
            )
            continue
        run_df = run_df.select(["molecule_chembl_id", "docking_score"]).with_columns(
            pl.lit(run_idx).alias("run_idx")
        )
        all_run_frames.append(run_df)

    if not all_run_frames:
        print(f"  [SKIP] {benchmark_name}: no variance run data found.")
        return

    stacked = pl.concat(all_run_frames)
    mean_scores = stacked.group_by("molecule_chembl_id").agg(
        pl.col("docking_score").mean().alias("mean_docking_score")
    )

    # ── Read bioactivity CSV and join ────────────────────────────────────────
    bio_df = pl.read_csv(csv_path)

    # Drop existing columns so the script is idempotent
    for col in ["mean_docking_score", "score"]:
        if col in bio_df.columns:
            bio_df = bio_df.drop(col)

    merged = bio_df.join(mean_scores, on="molecule_chembl_id", how="left")

    # ── Compute normalised score ─────────────────────────────────────────────
    if low_score is not None and high_score is not None:
        denom = low_score - high_score
        if abs(denom) > 1e-6:
            merged = merged.with_columns(
                ((pl.lit(low_score) - pl.col("mean_docking_score")) / denom).alias(
                    "score"
                )
            )
        else:
            merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))
    else:
        merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))

    # ── Write back ───────────────────────────────────────────────────────────
    merged.write_csv(csv_path)

    n_matched = merged.filter(pl.col("mean_docking_score").is_not_null()).height
    n_total = merged.height
    print(
        f"  [OK]  {benchmark_name}: matched {n_matched}/{n_total} compounds with "
        f"variance run docking data."
    )


def main() -> None:
    benchmarks = ["AKT1", "CHK1", "ITK", "PCK1", "TTK", "VEGFR2"]

    print(f"add_docking_baselines.py")
    print(f"  Variance runs dir : {VARIANCE_RUNS_DIR}")
    print(f"  Bioactivity dir   : {BIOACTIVITY_DIR}")
    print()

    if not VARIANCE_RUNS_DIR.exists():
        print(
            f"ERROR: variance_runs/ directory not found at {VARIANCE_RUNS_DIR}.",
            file=sys.stderr,
        )
        sys.exit(1)

    for bm in benchmarks:
        print(f"Processing {bm} ...")
        try:
            process_benchmark(bm)
        except Exception as exc:
            print(f"  [ERROR] {bm}: {exc}")

    print()
    print("Done. Re-run the script to refresh if variance run data changes.")


if __name__ == "__main__":
    main()
