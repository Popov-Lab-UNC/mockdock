#!/usr/bin/env python3
"""
Summarize docking benchmark results.

Accepts either:
  - A single run directory: benchmark_runs/run_YYYYMMDD_HHMMSS/
  - The benchmark_runs parent: benchmark_runs/  (processes all run_* subdirs)

Folder layout (per run):
  run_dir/
    benchmark_summary.csv    # Input: per-workflow rows from workflow.py
    run_summary.csv         # Output: enriched summary with success rates and metrics
    <target_id>_<pdb_id>/   # e.g. CHEMBL1075104_8TZC
      <doc_id>/             # e.g. CHEMBL5365501
        *_results.csv       # Used to compute Pearson/Spearman correlation metrics
        metrics.json        # Optional override (if present, used instead of computing)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None


def _compute_corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Return n_points, pearson, spearman, r2. Requires len >= 2 and non-zero variance."""
    out = {"n_points": 0, "pearson": np.nan, "spearman": np.nan, "r2": np.nan}
    if len(x) < 2 or len(y) < 2:
        return out
    out["n_points"] = int(len(x))
    if np.var(x) > 0 and np.var(y) > 0 and pearsonr is not None:
        p, _ = pearsonr(x, y)
        s, _ = spearmanr(x, y)
        out["pearson"] = float(p)
        out["spearman"] = float(s)
        out["r2"] = float(p**2)
    return out


def _metrics_from_results_csv(csv_path: Path) -> dict:
    """Compute best_any and rmsd_constrained metrics from a *_results.csv file."""
    metrics = {"best_any": {}, "rmsd_constrained": {}}
    try:
        res = pd.read_csv(csv_path)
    except Exception:
        return metrics
    if "pchembl_value" not in res.columns:
        return metrics
    res["pchembl_value"].dropna().to_numpy()

    # best_any: score_best_any vs pchembl_value (all rows with valid score)
    score_col = "score_best_any" if "score_best_any" in res.columns else "docking_score"
    x_all = res[score_col].replace(999.9, np.nan).dropna()
    valid = res.loc[x_all.index, "pchembl_value"].notna()
    idx = x_all.index[valid]
    if len(idx) >= 2:
        x_vals = res.loc[idx, score_col].to_numpy()
        y_vals = res.loc[idx, "pchembl_value"].to_numpy()
        metrics["best_any"] = _compute_corr_stats(x_vals, y_vals)

    # rmsd_constrained: docking_score vs pchembl_value (only valid_pose_found)
    if "valid_pose_found" in res.columns and "docking_score" in res.columns:
        valid_mask = res["valid_pose_found"].fillna(False).astype(bool)
        sub = res.loc[valid_mask, ["docking_score", "pchembl_value"]].dropna()
        if len(sub) >= 2:
            metrics["rmsd_constrained"] = _compute_corr_stats(
                sub["docking_score"].to_numpy(), sub["pchembl_value"].to_numpy()
            )

    return metrics


def summarize_run(run_dir: Path) -> bool:
    """Summarize a single benchmark run. Returns True on success."""
    summary_path = run_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        print(f"  Skipping {run_dir.name}: benchmark_summary.csv not found.")
        return False

    df = pd.read_csv(summary_path)

    # Success rates (avoid division by zero)
    d2 = df["n_compounds_standardized"].fillna(0)
    dr = df["n_compounds_docked"].fillna(0)
    df["success_rate_2d"] = np.where(
        d2 > 0, df["n_compounds_matched_2d"] / d2 * 100, 0
    ).round(2)
    df["success_rate_rmsd"] = np.where(dr > 0, df["n_valid_poses"] / dr * 100, 0).round(
        2
    )

    # Compute correlation metrics from *_results.csv (or metrics.json if present)
    # Layout: run_dir/<target_id>_<pdb_id>/<doc_id>/<{prefix}_results.csv
    # Or: run_dir/<target_id>_<pdb_id>/<doc_id>_<assay_id>/{prefix}_results.csv (if we ever change that)

    def get_key(row):
        k = f"{row['target_id']}_{row['pdb_id']}_{row['doc_id']}"
        if (
            "assay_id" in df.columns
            and pd.notna(row["assay_id"])
            and row["assay_id"] != ""
        ):
            k += f"_{row['assay_id']}"
        return k

    df["workflow_key"] = df.apply(get_key, axis=1)
    metrics_by_key = {}

    for mj in list(run_dir.glob("*/metrics.json")) + list(
        run_dir.glob("*/*/metrics.json")
    ):
        if "debug" in mj.parts:
            continue
        try:
            payload = json.loads(mj.read_text())
            tid, pid, did = (
                payload.get("target_id"),
                payload.get("pdb_id"),
                payload.get("doc_id"),
            )
            aid = payload.get("assay_id")
            if tid and pid and did:
                key = f"{tid}_{pid}_{did}"
                if aid:
                    key += f"_{aid}"
                metrics_by_key[key] = payload.get("metrics") or {}
        except Exception:
            continue

    for target_pdb in run_dir.iterdir():
        if not target_pdb.is_dir() or "debug" in target_pdb.parts:
            continue
        for doc_dir in target_pdb.iterdir():
            if not doc_dir.is_dir():
                continue

            # Find any _results.csv in this doc_dir
            for results_csv in doc_dir.glob("*_results.csv"):
                # Prefix is filename without _results.csv
                key = results_csv.stem.replace("_results", "")
                if key in metrics_by_key:
                    continue
                metrics_by_key[key] = _metrics_from_results_csv(results_csv)

    def _get_metric(workflow_key: str, metric_type: str, key: str):
        m = metrics_by_key.get(workflow_key) or {}
        d = m.get(metric_type) or {}
        return d.get(key)

    df["best_any_n_points"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "best_any", "n_points")
    )
    df["best_any_pearson"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "best_any", "pearson")
    )
    df["best_any_spearman"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "best_any", "spearman")
    )
    df["best_any_r2"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "best_any", "r2")
    )
    df["rmsd_n_points"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "rmsd_constrained", "n_points")
    )
    df["rmsd_pearson"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "rmsd_constrained", "pearson")
    )
    df["rmsd_spearman"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "rmsd_constrained", "spearman")
    )
    df["rmsd_r2"] = df["workflow_key"].apply(
        lambda w: _get_metric(w, "rmsd_constrained", "r2")
    )

    total = len(df)
    success = len(df[df["status"] == "SUCCESS"])
    failed = total - success
    pct_success = (success / total * 100) if total else 0
    pct_failed = (failed / total * 100) if total else 0

    print(f"\n--- {run_dir.name} ---")
    print(f"Total workflows: {total}")
    print(f"Successful:      {success} ({pct_success:.1f}%)")
    print(f"Failed:          {failed} ({pct_failed:.1f}%)")
    if failed > 0:
        counts = df[df["status"] != "SUCCESS"]["status"].value_counts()
        for status, count in counts.items():
            print(f"  - {status}: {count}")

    run_summary_path = run_dir / "run_summary.csv"
    df.to_csv(run_summary_path, index=False)
    print(f"Saved: {run_summary_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Summarize docking benchmark results.",
        epilog="Pass a single run dir (e.g. benchmark_runs/run_20260201_051320) or benchmark_runs/ to process all runs.",
    )
    parser.add_argument(
        "path",
        help="Benchmark run directory or benchmark_runs parent (e.g. benchmark_runs/run_20260201_051320 or benchmark_runs/)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    run_dirs = []
    if (path / "benchmark_summary.csv").exists():
        run_dirs = [path]
    else:
        # Assume parent of run_* directories
        run_dirs = sorted(path.glob("run_*/"))
        run_dirs = [
            d for d in run_dirs if d.is_dir() and (d / "benchmark_summary.csv").exists()
        ]
        if not run_dirs:
            print(
                f"Error: No run directories with benchmark_summary.csv found under {path}"
            )
            sys.exit(1)
        print(f"Found {len(run_dirs)} run(s) to summarize")

    for run_dir in run_dirs:
        summarize_run(run_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
