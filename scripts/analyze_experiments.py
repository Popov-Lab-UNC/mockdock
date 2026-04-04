#!/usr/bin/env python3
# scripts/analyze_experiments.py
"""
Aggregates FCGMB experiment results across models, targets, and seeds.
Generates master CSVs and publication-quality figures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

try:
    from fcgmb import FCGMBEvaluator
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from fcgmb import FCGMBEvaluator


def setup_plotting():
    """Configure seaborn/matplotlib for publication-quality output."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Bitstream Vera Sans", "Computer Modern Sans Serif", "Lucida Grande", "Verdana", "Geneva", "Lucid", "Arial", "Helvetica", "Avant Garde", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def discover_results(exps_dir: Path) -> list[tuple[str, str, str, Path]]:
    """
    Discover all results.csv files in the exps directory.
    Returns: list of (model, seed_id, target, results_csv_path)
    """
    results = []
    # Structure: exps/{model}/run_{seed_id}/{target}/results.csv
    for model_dir in exps_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir() or not (run_dir.name.startswith("run_") or run_dir.name.startswith("202")):
                continue
            # Some run directories might be named by timestamp, others have a suffix like _r01
            seed_id = run_dir.name
            
            for target_dir in run_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                target_name = target_dir.name
                
                csv_path = target_dir / "results.csv"
                if csv_path.exists():
                    results.append((model_name, seed_id, target_name, csv_path))
    
    return results


def process_all(results_list: list[tuple[str, str, str, Path]], scratch_dir: Path = None) -> pl.DataFrame:
    """
    Process each results.csv through FCGMBEvaluator.
    Returns a master DataFrame with all metrics.
    """
    all_data = []
    
    # Cache for unique evaluator instances
    evaluators = {}
    
    for model, seed, target, csv_path in results_list:
        print(f"Evaluating {model} | {target} | {seed}...")
        
        if target not in evaluators:
            evaluators[target] = FCGMBEvaluator(target, scratch_dir=scratch_dir)
            
        try:
            # Check if metrics already computed to save time
            metrics_json = csv_path.parent / "eval_metrics.json"
            if metrics_json.exists():
                with open(metrics_json, "r") as f:
                    metrics = json.load(f)
            else:
                metrics = evaluators[target].compute_metrics(csv_path)
            
            # Flat row data
            row = {
                "model": model,
                "seed": seed,
                "target": target,
            }
            # Add metrics (minus descriptions)
            for k, v in metrics.items():
                if k != "descriptions" and not isinstance(v, dict):
                    row[k] = v
            
            all_data.append(row)
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            
    return pl.DataFrame(all_data)


def compute_aggregates(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute mean and std across seeds, and macro-average across targets."""
    # 1. Summary by (model, target) - aggregating over seeds
    metric_cols = [c for c in df.columns if c not in ["model", "seed", "target"]]
    
    agg_exprs = []
    for c in metric_cols:
        agg_exprs.append(pl.mean(c).alias(f"{c}_mean"))
        agg_exprs.append(pl.std(c).alias(f"{c}_std"))
        
    summary_df = df.group_by(["model", "target"]).agg(agg_exprs).sort(["model", "target"])
    
    # 2. Macro-average across targets for each model
    macro_agg_exprs = []
    for c in metric_cols:
        macro_agg_exprs.append(pl.mean(f"{c}_mean").alias(f"{c}_mean"))
        # std of the means across targets? Or mean of the stds? 
        # Usually macro-average just reports mean across targets.
    
    macro_df = summary_df.group_by("model").agg(macro_agg_exprs).sort("model")
    
    return summary_df, macro_df


# ─── Plotting Functions ─────────────────────────────────────────────

def plot_figure_1_bars(summary_df: pl.DataFrame, output_dir: Path):
    """Figure 1: Bar charts breakdown for each target using FacetGrids."""
    setup_plotting()
    metrics = ["avg_top_10", "validity", "novelty", "internal_diversity", "fragment_incorporation"]
    
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary_df.columns:
            continue
            
        print(f"  Generating breakdown bars for {metric}...")
        
        # Convert to pandas for FacetGrid
        pdf = summary_df.select(["model", "target", mean_col, std_col]).to_pandas()
        
        # Clean up column names for plotting
        display_name = metric.replace("_", " ").capitalize()
        pdf.rename(columns={mean_col: display_name}, inplace=True)
        
        g = sns.FacetGrid(pdf, col="target", col_wrap=3, height=5, aspect=1.3, sharey=True)
        g.map_dataframe(sns.barplot, x="model", y=display_name, hue="model", palette="viridis", legend=False)
        
        # Add error bars manually on each axis
        for ax, (_, target_data) in zip(g.axes.flat, pdf.groupby("target")):
            models = target_data["model"].unique()
            x_pos = np.arange(len(models))
            ax.errorbar(x=x_pos, y=target_data[display_name], yerr=target_data[std_col], 
                        fmt='none', c='black', capsize=5)
            
        g.set_xticklabels(rotation=45)
        g.set_titles("{col_name}")
        g.fig.suptitle(f"Benchmark Performance: {display_name}", y=1.02, fontsize=18)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"fig1_bars_{metric}.svg")
        plt.savefig(output_dir / f"fig1_bars_{metric}.png", dpi=300)
        plt.close()


def plot_figure_2_trajectory(exps_dir: Path, output_dir: Path, k: int = 10):
    """Figure 2: Running top-k average with 90% CI shaded band for each target."""
    setup_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover all unique targets
    targets_found = set()
    for model_dir in exps_dir.iterdir():
        if not model_dir.is_dir(): continue
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir(): continue
            for target_dir in run_dir.iterdir():
                if target_dir.is_dir() and (target_dir / "results.csv").exists():
                    targets_found.add(target_dir.name)
    
    targets_found = sorted(list(targets_found))
    
    for target in targets_found:
        print(f"  Generating trajectory for {target}...")
        plt.figure(figsize=(12, 7))
        
        # Group results by model for THIS target
        model_results = {}
        for model_dir in exps_dir.iterdir():
            if not model_dir.is_dir(): continue
            model_name = model_dir.name
            paths = []
            for run_dir in model_dir.iterdir():
                csv = run_dir / target / "results.csv"
                if csv.exists():
                    paths.append(csv)
            if paths:
                model_results[model_name] = paths

        if not model_results: continue
        
        palette = sns.color_palette("husl", len(model_results))
        
        for i, (model, csv_paths) in enumerate(model_results.items()):
            curves = []
            max_len = 1000 
            
            for csv in csv_paths:
                df = pl.read_csv(csv)
                scores = df["normalized_score"].to_list()
                buffer = []
                curr_curve = []
                for s in scores:
                    buffer.append(s)
                    top_k = sorted(buffer, reverse=True)[:k]
                    curr_curve.append(np.mean(top_k))
                
                if len(curr_curve) < max_len:
                    last_val = curr_curve[-1] if curr_curve else 0.0
                    curr_curve.extend([last_val] * (max_len - len(curr_curve)))
                else:
                    curr_curve = curr_curve[:max_len]
                curves.append(curr_curve)
                
            curves = np.array(curves)
            n = len(curves)
            mean_curve = np.mean(curves, axis=0)
            sem_curve = np.std(curves, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(max_len)
            
            t_crit = stats.t.ppf(0.95, df=max(1, n-1))
            ci_half = t_crit * sem_curve
            
            x = np.arange(1, max_len + 1)
            plt.plot(x, mean_curve, label=model, lw=2.5, color=palette[i])
            plt.fill_between(x, mean_curve - ci_half, mean_curve + ci_half, alpha=0.2, color=palette[i])
            
        plt.xlabel("Cumulative Oracle Calls")
        plt.ylabel(f"Running Avg Top-{k} Normalized Score")
        plt.title(f"Optimization Trajectory: {target} (90% CI, k={k})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig2_trajectory_{target}.svg")
        plt.savefig(output_dir / f"fig2_trajectory_{target}.png", dpi=300)
        plt.close()


def plot_figure_3_radar(macro_df: pl.DataFrame, output_dir: Path):
    """Figure 3: Radar chart of macro-averaged metrics."""
    setup_plotting()
    
    # Key metrics for radar
    metrics_map = {
        "avg_top_10_mean": "Docking (Top-10)",
        "validity_mean": "Validity",
        "novelty_mean": "Novelty",
        "internal_diversity_mean": "Diversity",
        "fragment_incorporation_mean": "Fragment",
        "oracle_efficiency_80_mean": "Efficiency"
    }
    
    # 1. Invert and normalize efficiency (assuming max 1000 oracle calls)
    data_df = macro_df.select(["model"] + list(metrics_map.keys()))
    if "oracle_efficiency_80_mean" in data_df.columns:
        data_df = data_df.with_columns(
            (1.0 - (pl.col("oracle_efficiency_80_mean").clip(0, 1000) / 1000.0)).alias("oracle_efficiency_80_mean")
        )

    # 2. Extract data for plotting
    labels = list(metrics_map.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Color palette for models
    models = data_df["model"].to_list()
    palette = sns.color_palette("husl", len(models))
    
    for i, row in enumerate(data_df.to_pandas().itertuples()):
        values = [getattr(row, m) for m in metrics_map.keys()]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row.model, color=palette[i])
        ax.fill(angles, values, alpha=0.1, color=palette[i])
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    # Set radial limits and grid
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(180 / num_vars)
    
    plt.title(f"Macro-Averaged Model Capability Profile", y=1.08, fontsize=18)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_radar_metrics.svg")
    plt.savefig(output_dir / "fig3_radar_metrics.png", dpi=300)
    plt.close()


def plot_figure_4_heatmap(macro_df: pl.DataFrame, output_dir: Path):
    """Figure 4: Heatmap of Model x Metric."""
    cols_to_plot = [c for c in macro_df.columns if c.endswith("_mean")]
    plot_df = macro_df.select(["model"] + cols_to_plot).to_pandas().set_index("model")
    
    # Strip "_mean" from col labels
    plot_df.columns = [c.replace("_mean", "").replace("_", " ").capitalize() for c in plot_df.columns]
    
    # Select subset of most interesting metrics
    interesting = ["Avg top 10", "Validity", "Novelty", "Internal diversity", "Scaffold diversity", "Mean qed", "Fraction lipinski", "Fraction pains free"]
    cols = [c for c in interesting if c in plot_df.columns]
    plot_df = plot_df[cols]
    
    # Normalize per column for color scale
    norm_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(norm_df, annot=plot_df, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Normalized Performance'})
    plt.title("Master Performance Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_heatmap.svg")
    plt.savefig(output_dir / "fig4_heatmap.png", dpi=300)
    plt.close()


def plot_figure_5_small_multiples(summary_df: pl.DataFrame, output_dir: Path):
    """Figure 5: Per-target breakdown bar charts."""
    plt.figure(figsize=(16, 10))
    # Filter for standard metrics
    plot_df = summary_df.select(["model", "target", "avg_top_10_mean", "avg_top_10_std"]).to_pandas()
    
    g = sns.FacetGrid(plot_df, col="target", col_wrap=3, height=4, aspect=1.2)
    g.map_dataframe(sns.barplot, x="model", y="avg_top_10_mean", hue="model", palette="viridis", legend=False)
    
    # Adding error bars is tricky in facetgrid barplot, skipping for MVP or manual loop
    
    g.set_xticklabels(rotation=45)
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_facet_targets.svg")
    plt.savefig(output_dir / "fig5_facet_targets.png", dpi=300)
    plt.close()


def plot_figure_8_quality(macro_df: pl.DataFrame, output_dir: Path):
    """Figure 8: Fraction Lipinski and PAINS-free."""
    metrics = ["fraction_lipinski_mean", "fraction_pains_free_mean"]
    if not all(m in macro_df.columns for m in metrics):
        return

    plot_df = macro_df.select(["model"] + metrics).to_pandas().melt(id_vars="model")
    plot_df["variable"] = plot_df["variable"].str.replace("_mean", "").str.replace("_", " ").str.capitalize()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="model", y="value", hue="variable")
    plt.title("Chemical Quality Comparison")
    plt.ylabel("Fraction")
    plt.ylim(0, 1.05)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_dir / "fig8_quality_summary.svg")
    plt.savefig(output_dir / "fig8_quality_summary.png", dpi=300)
    plt.close()


# ─── Main Logic ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze FCGMB experimental results.")
    parser.add_argument("--exps-dir", type=Path, default=Path("exps"), help="Path to exps folder")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Path for aggregated outputs")
    parser.add_argument("--scratch-dir", type=Path, default=None, help="FCGMB scratch dir")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    setup_plotting()
    
    # 1. Discover
    results_list = discover_results(args.exps_dir)
    print(f"Discovered {len(results_list)} results files.")
    
    if not results_list:
        print("No results to analyze.")
        return

    # 2. Process
    full_df = process_all(results_list, scratch_dir=args.scratch_dir)
    full_df.write_csv(args.output_dir / "metrics_all.csv")
    
    # 3. Aggregate
    summary_df, macro_df = compute_aggregates(full_df)
    summary_df.write_csv(args.output_dir / "metrics_summary.csv")
    macro_df.write_csv(args.output_dir / "metrics_summary_macro.csv")
    
    # 4. Figure generation
    fig_dir = args.output_dir / "figures"
    print("Generating Figure 1 (Bars)...")
    plot_figure_1_bars(summary_df, fig_dir)
    
    print("Generating Figure 2 (Trajectories)...")
    plot_figure_2_trajectory(args.exps_dir, fig_dir)
    
    print("Generating Figure 3 (Radar)...")
    plot_figure_3_radar(macro_df, fig_dir)
    
    print("Generating Figure 4 (Heatmap)...")
    plot_figure_4_heatmap(macro_df, fig_dir)
    
    print("Generating Figure 5 (Small Multiples)...")
    plot_figure_5_small_multiples(summary_df, fig_dir)
    
    print("Generating Figure 8 (Quality)...")
    plot_figure_8_quality(macro_df, fig_dir)
    
    print(f"\nDone! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
