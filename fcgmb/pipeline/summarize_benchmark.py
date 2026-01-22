#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path
import sys
import json

def main():
    parser = argparse.ArgumentParser(description="Summarize docking benchmark results")
    parser.add_argument("run_dir", help="Path to the benchmark run directory (e.g., benchmark_runs/run_20260108_143022)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "benchmark_summary.csv"

    if not summary_path.exists():
        print(f"Error: Summary file {summary_path} not found.")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    df["success_rate_2d"] = (df["n_compounds_matched_2d"] / df["n_compounds_standardized"]).replace([pd.NA], 0)
    df["success_rate_rmsd"] = (df["n_valid_poses"] / df["n_compounds_docked"]).replace([pd.NA], 0)
    df["success_rate_2d"] = (df["success_rate_2d"] * 100).round(2)
    df["success_rate_rmsd"] = (df["success_rate_rmsd"] * 100).round(2)

    # Load per-workflow correlation metrics from metrics.json (if present)
    # We store best-any and RMSD-constrained metrics as columns in the same table.
    #
    # New layout (recommended):
    #   run_dir/<target_id>_<pdb_id>/<doc_id>/metrics.json
    # Legacy layout:
    #   run_dir/<target_id>_<pdb_id>/metrics.json
    #
    # Index by a stable key derived from the payload: <target_id>_<pdb_id>_<doc_id>
    metrics_by_key = {}
    metric_paths = []
    metric_paths.extend(list(run_dir.glob("*/metrics.json")))
    metric_paths.extend(list(run_dir.glob("*/*/metrics.json")))

    for mj in metric_paths:
        # Skip debug folder (and anything under it)
        if "debug" in mj.parts:
            continue
        try:
            payload = json.loads(mj.read_text())
            tid = payload.get("target_id")
            pid = payload.get("pdb_id")
            did = payload.get("doc_id")
            if tid and pid and did:
                metrics_by_key[f"{tid}_{pid}_{did}"] = payload.get("metrics") or {}
        except Exception:
            continue

    def _get_metric(workflow_key: str, metric_type: str, key: str):
        m = metrics_by_key.get(workflow_key) or {}
        d = m.get(metric_type) or {}
        return d.get(key)

    # workflow_key is derived from target_id + pdb_id + doc_id in run_workflow.py
    df["workflow_key"] = df["target_id"].astype(str) + "_" + df["pdb_id"].astype(str) + "_" + df["doc_id"].astype(str)

    # Best-any metrics
    df["best_any_n_points"] = df["workflow_key"].apply(lambda w: _get_metric(w, "best_any", "n_points"))
    df["best_any_pearson"] = df["workflow_key"].apply(lambda w: _get_metric(w, "best_any", "pearson"))
    df["best_any_spearman"] = df["workflow_key"].apply(lambda w: _get_metric(w, "best_any", "spearman"))
    df["best_any_r2"] = df["workflow_key"].apply(lambda w: _get_metric(w, "best_any", "r2"))

    # RMSD-constrained metrics
    df["rmsd_n_points"] = df["workflow_key"].apply(lambda w: _get_metric(w, "rmsd_constrained", "n_points"))
    df["rmsd_pearson"] = df["workflow_key"].apply(lambda w: _get_metric(w, "rmsd_constrained", "pearson"))
    df["rmsd_spearman"] = df["workflow_key"].apply(lambda w: _get_metric(w, "rmsd_constrained", "spearman"))
    df["rmsd_r2"] = df["workflow_key"].apply(lambda w: _get_metric(w, "rmsd_constrained", "r2"))

    print("\n" + "="*40)
    print("=== Benchmark Run Summary ===")
    print("="*40)
    
    total = len(df)
    success = len(df[df['status'] == 'SUCCESS'])
    failed = total - success
    
    print(f"Total workflows: {total}")
    print(f"Successful:      {success} ({success/total*100:.1f}%)")
    print(f"Failed:          {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        print("\nFailure Breakdown:")
        counts = df[df['status'] != 'SUCCESS']['status'].value_counts()
        for status, count in counts.items():
            print(f"  - {status}: {count}")

    # Write single run summary CSV
    run_summary_path = run_dir / "run_summary.csv"
    df.to_csv(run_summary_path, index=False)
    print(f"\nRun summary saved to: {run_summary_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
