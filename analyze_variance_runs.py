#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from fcgmb.variance import get_pactivity


def _iter_run_dirs(runs_dir: Path) -> List[Path]:
    return sorted([p for p in runs_dir.glob("run_*") if p.is_dir()])


def _iter_results(run_dir: Path) -> List[Tuple[str, Path]]:
    results = []
    for csv_path in run_dir.glob("**/*_results_full.csv"):
        # Layout: run_dir/<target_id>_<pdb_id>/<doc_id>/*_results_full.csv
        try:
            target_pdb = csv_path.parent.parent.name
            doc_id = csv_path.parent.name
            system_key = f"{target_pdb}_{doc_id}"
            results.append((system_key, csv_path))
        except Exception:
            continue
    return results


def _valid_score_mask(df: pl.DataFrame, score_cols: List[str]) -> pl.Expr:
    exprs = [(pl.col(c).is_not_null()) & (pl.col(c).is_finite()) & (pl.col(c) < 900) for c in score_cols]
    return pl.fold(pl.lit(True), lambda acc, x: acc & x, exprs)


def _compute_system_correlations(df: pl.DataFrame, config_path: Path) -> Tuple[float, float]:
    if df.is_empty() or "docking_score" not in df.columns:
        return float("nan"), float("nan")
    p_activities, _ = get_pactivity(df, config_path)
    scores = df.get_column("docking_score").to_numpy()
    mask = np.isfinite(scores) & np.isfinite(p_activities) & (scores < 900)
    if np.sum(mask) < 2:
        return float("nan"), float("nan")
    pearson = pearsonr(scores[mask], p_activities[mask])[0]
    spearman = spearmanr(scores[mask], p_activities[mask]).correlation
    return pearson, spearman


def plot_run_barplot(
    runs_dir: Path,
    config_dir: Path,
    output_dir: Path
) -> Path:
    run_dirs = _iter_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_dir}")

    system_corrs: Dict[str, List[float]] = {}
    for run_dir in run_dirs:
        for system_key, csv_path in _iter_results(run_dir):
            config_path = config_dir / f"{system_key}.yaml"
            if not config_path.exists():
                continue
            df = pl.read_csv(csv_path)
            pearson, spearman = _compute_system_correlations(df, config_path)
            if np.isfinite(pearson):
                system_corrs.setdefault(system_key, []).append((pearson, spearman))

    system_stats = []
    for system_key, corrs in sorted(system_corrs.items()):
        pearsons = [c[0] for c in corrs]
        spearmans = [c[1] for c in corrs if np.isfinite(c[1])]
        mean_corr = float(np.mean(pearsons)) if pearsons else float("nan")
        std_corr = float(np.std(pearsons)) if pearsons else float("nan")
        r2_mean = float(np.mean(np.square(pearsons))) if pearsons else float("nan")
        r2_std = float(np.std(np.square(pearsons))) if pearsons else float("nan")
        spearman_mean = float(np.mean(spearmans)) if spearmans else float("nan")
        spearman_std = float(np.std(spearmans)) if spearmans else float("nan")
        system_stats.append(
            {
                "system": system_key,
                "n_runs": len(pearsons),
                "pearson_mean": mean_corr,
                "pearson_std": std_corr,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "spearman_mean": spearman_mean,
                "spearman_std": spearman_std,
            }
        )

    stats_df = pl.DataFrame(system_stats).sort("pearson_mean")
    stats_csv = output_dir / "run_correlation_summary.csv"
    stats_df.write_csv(stats_csv)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    x = np.arange(len(system_stats))
    means = [r["pearson_mean"] for r in system_stats]
    stds = [r["pearson_std"] for r in system_stats]
    labels = [r["system"] for r in system_stats]
    plt.bar(x, means, yerr=stds, capsize=4, color="#4c72b0", alpha=0.85)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mean Pearson (Docking Score vs pActivity)")
    plt.title("Per-System Mean/Std of Run Correlations")
    out_path = output_dir / "system_mean_std_barplot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_system_variance(
    runs_dir: Path,
    config_dir: Path,
    output_dir: Path
) -> List[Path]:
    run_dirs = _iter_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_dir}")

    # Collect per-system data across runs
    system_data: Dict[str, List[pl.DataFrame]] = {}
    for run_dir in run_dirs:
        for system_key, csv_path in _iter_results(run_dir):
            df = pl.read_csv(csv_path)
            system_data.setdefault(system_key, []).append(df)

    plot_paths = []
    for system_key, df_list in system_data.items():
        if len(df_list) < 2:
            continue

        merged = None
        for i, df in enumerate(df_list):
            columns = ["canonical_smiles", "docking_score"]
            if "standard_value" in df.columns:
                columns.append("standard_value")
            if "pchembl_value" in df.columns:
                columns.append("pchembl_value")
            subset = df.select(columns).rename({"docking_score": f"score_{i}"})
            if merged is None:
                merged = subset
            else:
                drop_cols = [c for c in ["standard_value", "pchembl_value"] if c in merged.columns]
                merged = merged.join(subset.drop(drop_cols), on="canonical_smiles", how="inner")

        if merged is None or merged.is_empty():
            continue

        score_cols = [c for c in merged.columns if c.startswith("score_")]
        valid_mask = _valid_score_mask(merged, score_cols)
        clean = merged.filter(valid_mask)
        if clean.is_empty():
            continue

        scores_matrix = clean.select(score_cols).to_numpy()
        means = np.mean(scores_matrix, axis=1)
        stds = np.std(scores_matrix, axis=1)

        config_path = config_dir / f"{system_key}.yaml"
        if not config_path.exists():
            continue
        p_activities, activity_label = get_pactivity(clean, config_path)

        plt.figure(figsize=(10, 7))
        sns.set_style("whitegrid")
        plt.errorbar(
            means,
            p_activities,
            xerr=stds,
            fmt="o",
            color="#2c7bb6",
            ecolor="#d7191c",
            alpha=0.6,
            capsize=3,
            markersize=5,
        )
        plt.title(f"Docking Score Variance vs {activity_label}\nSystem: {system_key}")
        plt.xlabel("Mean Docking Score (kcal/mol)")
        plt.ylabel(activity_label)

        out_path = output_dir / f"{system_key}_variance_plot.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(out_path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(description="Analyze variance runs across multiple run_* directories")
    parser.add_argument("--runs-dir", type=str, default="variance_runs", help="Base variance runs directory")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory with system YAML configs")
    parser.add_argument("--output-dir", type=str, default="variance_plots", help="Output directory for plots")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    barplot_path = plot_run_barplot(runs_dir, config_dir, output_dir)
    plot_system_variance(runs_dir, config_dir, output_dir)

    print(f"Saved bar plot to: {barplot_path}")
    print(f"Saved per-system variance plots to: {output_dir}")


if __name__ == "__main__":
    main()
