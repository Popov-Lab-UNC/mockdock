#!/usr/bin/env python3
"""add_docking_baselines.py – Phase 2 one-time helper.

Reads the variance runs and computes a
mean docking score for each compound across the repeats.  The result is
written back into the six canonical bioactivity CSVs
(mockdock/bioactivity_data/<benchmark>.csv) as three score columns:

  mean_docking_score  – raw mean docking score in kcal/mol (lower = better)
  norm_score          – uncapped normalised score using each benchmark's
                        low_score / high_score calibration points. Formula:
                        (low_score - mean_docking_score) / (low_score - high_score).
                        Can be negative or exceed 1.0 when mean_docking_score
                        falls outside the calibration range.
  reward_score        – norm_score clipped to [0, 1], used as the RL reward.
  score               – backward-compatible alias of reward_score.

low_score and high_score are computed from the variance run data: high_score is
the best (minimum) mean_docking_score observed; low_score is the worst (maximum)
mean_docking_score observed. These values are written back into the benchmark
YAML configs (mockdock/configs/<benchmark>.yaml).

Run once from the benchmark root::

    python add_docking_baselines.py

The script is idempotent: if the columns already exist they are overwritten.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import yaml
import tomllib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
VARIANCE_RUNS_DIR = Path(
    os.environ.get(
        "MOCKDOCK_VARIANCE_RUNS_DIR",
        BENCHMARK_ROOT.parent / "mockdock_data" / "variance_runs",
    )
)
BIOACTIVITY_DIR = BENCHMARK_ROOT / "src" / "mockdock" / "bioactivity_data"
CONFIGS_DIR = BENCHMARK_ROOT / "src" / "mockdock" / "configs"


# Map each canonical benchmark name to its (target_id, pdb_id, doc_id, assay_id)
# so we can reconstruct the path to each variant run results CSV.
def _load_benchmark_config(benchmark_name: str) -> dict:
    config_path = CONFIGS_DIR / f"{benchmark_name}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _find_results_csv(run_dir: Path, target_id: str, pdb_id: str, doc_id: str) -> Path | None:
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
        candidates = list(subdir.glob("*_results.csv"))
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

    # ── Collect per-compound docking scores from all variance runs ──────────
    all_run_frames: list[pl.DataFrame] = []
    run_dirs = sorted(p for p in VARIANCE_RUNS_DIR.glob("run_*") if p.is_dir())
    for run_idx, run_dir in enumerate(run_dirs, start=1):
        results_csv = _find_results_csv(run_dir, target_id, pdb_id, doc_id)
        if results_csv is None:
            print(f"  [WARN] {benchmark_name}: no results CSV in {run_dir.name}, skipping.")
            continue

        run_df = pl.read_csv(results_csv)
        # Keep only chembl ID and docking score; tag the run number
        if "molecule_chembl_id" not in run_df.columns or "docking_score" not in run_df.columns:
            print(
                f"  [WARN] {benchmark_name}: {run_dir.name} CSV missing expected columns, skipping."
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
    for col in ["mean_docking_score", "norm_score", "reward_score", "score"]:
        if col in bio_df.columns:
            bio_df = bio_df.drop(col)
    merged = bio_df.join(mean_scores, on="molecule_chembl_id", how="left")

    # ── Compute low_score and high_score from variance run data ──────────────
    # high_score = best (min) mean_docking_score; low_score = worst (max) mean_docking_score.
    with_scores = merged.filter(pl.col("mean_docking_score").is_not_null())
    if with_scores.is_empty():
        print(f"  [SKIP] {benchmark_name}: no compounds with variance run data.")
        return

    high_score = float(with_scores["mean_docking_score"].min())
    low_score = float(with_scores["mean_docking_score"].max())

    # Write calibration values back to TOML config
    cfg["low_score"] = round(low_score, 3)
    cfg["high_score"] = round(high_score, 3)
    config_path = CONFIGS_DIR / f"{benchmark_name}.toml"
    with open(config_path, "w") as f:
        for k, v in cfg.items():
            if isinstance(v, bool):
                f.write(f"{k} = {str(v).lower()}\n")
            elif isinstance(v, (int, float)):
                f.write(f"{k} = {v}\n")
            elif isinstance(v, str):
                f.write(f'{k} = "{v}"\n')

    # ── Compute scores ───────────────────────────────────────────────────────
    denom = low_score - high_score
    if abs(denom) > 1e-6:
        merged = (
            merged.with_columns(
                ((pl.lit(low_score) - pl.col("mean_docking_score")) / denom).alias("norm_score")
            )
            .with_columns(pl.col("norm_score").clip(0.0, 1.0).alias("reward_score"))
            .with_columns(pl.col("reward_score").alias("score"))
        )
    else:
        merged = merged.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("norm_score"),
            pl.lit(None, dtype=pl.Float64).alias("reward_score"),
            pl.lit(None, dtype=pl.Float64).alias("score"),
        )

    # ── Write back ───────────────────────────────────────────────────────────
    merged.write_csv(csv_path)

    n_matched = merged.filter(pl.col("mean_docking_score").is_not_null()).height
    n_total = merged.height
    print(
        f"  [OK]  {benchmark_name}: matched {n_matched}/{n_total} compounds with "
        f"variance run docking data."
    )


def main() -> None:
    benchmarks = sorted(p.stem for p in CONFIGS_DIR.glob("*.toml"))

    print("add_docking_baselines.py")
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
