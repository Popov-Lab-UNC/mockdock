#!/usr/bin/env python3
# scripts/analyze_experiments.py
"""
Aggregates mockdock experiment results across models, targets, and seeds.
Generates master CSVs and publication-quality figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

try:
    from mockdock import MDEvaluator
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mockdock import MDEvaluator


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
    Process each results.csv through MDEvaluator.
    Returns a master DataFrame with all metrics.
    """
    all_data = []
    
    # Cache for unique evaluator instances
    evaluators = {}
    
    for model, seed, target, csv_path in results_list:
        print(f"Evaluating {model} | {target} | {seed}...")
        
        if target not in evaluators:
            evaluators[target] = MDEvaluator(target, scratch_dir=scratch_dir)
            
        try:
            # Check if metrics already computed to save time
            metrics_json = csv_path.parent / "eval_metrics.json"
            if metrics_json.exists():
                with open(metrics_json) as f:
                    metrics = json.load(f)
            else:
                metrics = evaluators[target].compute_metrics(csv_path)
            
            # Optional runtime metadata from oracle-side metrics.json
            runtime_path = csv_path.parent / "metrics.json"
            runtime_metrics = _read_runtime_metrics(runtime_path)

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
            row.update(runtime_metrics)
            
            all_data.append(row)
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            
    return pl.DataFrame(all_data)


def _read_runtime_metrics(metrics_json_path: Path) -> dict:
    """Extract runtime fields from run-level metrics.json when present."""
    if not metrics_json_path.exists():
        return {}
    try:
        with open(metrics_json_path) as f:
            payload = json.load(f)
    except Exception:
        return {}

    def _get_value(key: str):
        return payload.get(key)

    out = {}
    aliases = {
        "n_molecules_total": ["n_molecules_total"],
        "n_molecules_attempted": ["n_molecules_attempted"],
        "total_gen_time": ["total_gen_time"],
        "avg_gen_time_per_mol": ["avg_gen_time_per_mol"],
        "total_eval_time": ["total_eval_time"],
        "avg_eval_time_per_mol": ["avg_eval_time_per_mol"],
        "total_time": ["total_time"],
        "avg_time_per_mol": ["avg_time_per_mol"],
    }
    for canonical_key, candidates in aliases.items():
        for candidate in candidates:
            value = _get_value(candidate)
            if value is not None:
                out[canonical_key] = value
                break
    return out


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

def _plot_metric_panels(
    full_df: pl.DataFrame,
    metrics_map: dict[str, str],
    output_path_stem: Path,
    figure_title: str,
    ncols: int = 3,
):
    """Shared panel plot helper: one panel per metric with benchmark-wise comparisons."""
    setup_plotting()
    available = [(metric, label) for metric, label in metrics_map.items() if metric in full_df.columns]
    if not available:
        return

    pdf = full_df.select(["model", "target"] + [m for m, _ in available]).to_pandas()
    targets = sorted(pdf["target"].unique())
    n_panels = len(available)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (metric, label) in enumerate(available):
        ax = axes_flat[idx]
        sns.pointplot(
            data=pdf,
            x="target",
            y=metric,
            hue="model",
            order=targets,
            estimator=np.mean,
            errorbar=("ci", 95),
            dodge=0.4,
            markers="o",
            linestyles="-",
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=35)
        if idx > 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(6, len(labels)))
    fig.suptitle(figure_title, y=1.06, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path_stem.with_suffix(".svg"))
    fig.savefig(output_path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_figure_1_generation(full_df: pl.DataFrame, output_dir: Path):
    """Figure 1: Intrinsic/extrinsic generation quality across benchmarks."""
    metrics_map = {
        "validity": "Validity",
        "uniqueness": "Uniqueness",
        "fragment_incorporation": "Fragment 2D",
        "novelty": "Novelty",
        "nonidenticality": "Nonidenticality",
        "effective_novelty": "Effective Novelty",
    }
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig1_generation_metrics",
        figure_title="Generation Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=3,
    )


def plot_figure_2_optimization(full_df: pl.DataFrame, output_dir: Path):
    """Figure 2: Optimization endpoints across benchmarks."""
    metrics_map = {
        "avg_top_10": "Avg Top-10 (Raw Score)",
        "avg_top_100": "Avg Top-100 (Raw Score)",
        "oracle_efficiency_80": "Oracle Efficiency @80% (optional)",
        "valid_pose_rate": "Valid Pose Rate (optional)",
    }
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig2_optimization_metrics",
        figure_title="Optimization Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=2,
    )


def plot_figure_3_quality(full_df: pl.DataFrame, output_dir: Path):
    """Figure 3: Medicinal chemistry quality, prioritizing novel compounds when available."""
    metrics_map = {
        "mean_qed_novel": "Mean QED (Novel Only)",
        "mean_sa_novel": "Mean SA (Novel Only)",
        "fraction_lipinski": "Fraction Lipinski",
        "fraction_pains_free": "Fraction PAINS-free",
    }
    fallback_map = {
        "mean_qed": "Mean QED (All Unique)",
        "mean_sa": "Mean SA (All Unique)",
    }
    if ("mean_qed_novel" not in full_df.columns) and ("mean_qed" in full_df.columns):
        metrics_map["mean_qed"] = fallback_map["mean_qed"]
    if ("mean_sa_novel" not in full_df.columns) and ("mean_sa" in full_df.columns):
        metrics_map["mean_sa"] = fallback_map["mean_sa"]
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig3_quality_metrics",
        figure_title="Chemical Quality Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=2,
    )


def plot_figure_4_trajectory(exps_dir: Path, output_dir: Path, k: int = 10):
    """Figure 4: Running top-k trajectories with median and IQR across seeds."""
    setup_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_found = set()
    for model_dir in exps_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for target_dir in run_dir.iterdir():
                if target_dir.is_dir() and (target_dir / "results.csv").exists():
                    targets_found.add(target_dir.name)
    targets_found = sorted(list(targets_found))

    for target in targets_found:
        print(f"  Generating trajectory for {target}...")
        plt.figure(figsize=(12, 7))
        model_results: dict[str, list[Path]] = {}
        for model_dir in exps_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            paths = []
            for run_dir in model_dir.iterdir():
                csv = run_dir / target / "results.csv"
                if csv.exists():
                    paths.append(csv)
            if paths:
                model_results[model_name] = paths
        if not model_results:
            continue

        palette = sns.color_palette("husl", len(model_results))
        for i, (model, csv_paths) in enumerate(model_results.items()):
            curves = []
            for csv in csv_paths:
                df = pl.read_csv(csv)
                scores = df["normalized_score"].to_list()
                running = []
                buffer = []
                for score in scores:
                    buffer.append(score)
                    running.append(float(np.mean(sorted(buffer, reverse=True)[:k])))
                if running:
                    curves.append(running)
            if not curves:
                continue

            min_len = min(len(curve) for curve in curves)
            if min_len < 2:
                continue
            aligned = np.array([curve[:min_len] for curve in curves])
            median_curve = np.median(aligned, axis=0)
            q25 = np.percentile(aligned, 25, axis=0)
            q75 = np.percentile(aligned, 75, axis=0)
            x = np.arange(1, min_len + 1)
            plt.plot(x, median_curve, label=model, lw=2.2, color=palette[i])
            plt.fill_between(x, q25, q75, alpha=0.2, color=palette[i])

        plt.xlabel("Cumulative Oracle Calls")
        plt.ylabel(f"Running Avg Top-{k} Normalized Score")
        plt.title(f"Optimization Trajectory: {target} (Median with IQR, k={k})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig4_trajectory_{target}.svg")
        plt.savefig(output_dir / f"fig4_trajectory_{target}.png", dpi=300)
        plt.close()


def write_table_1_macro_summary(macro_df: pl.DataFrame, output_dir: Path):
    """Table 1: macro summary grouped by generation/optimization/quality/runtime."""
    preferred_cols = [
        "model",
        "validity_mean",
        "uniqueness_mean",
        "fragment_incorporation_mean",
        "novelty_mean",
        "nonidenticality_mean",
        "effective_novelty_mean",
        "avg_top_10_mean",
        "avg_top_100_mean",
        "mean_qed_novel_mean",
        "mean_sa_novel_mean",
        "fraction_lipinski_mean",
        "fraction_pains_free_mean",
        "avg_gen_time_per_mol_mean",
        "avg_eval_time_per_mol_mean",
        "avg_time_per_mol_mean",
    ]
    existing_cols = [c for c in preferred_cols if c in macro_df.columns]
    if not existing_cols:
        return
    macro_df.select(existing_cols).write_csv(output_dir / "table1_macro_summary.csv")


# ─── Main Logic ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze mockdock experimental results.")
    parser.add_argument("--exps-dir", type=Path, default=Path("exps"), help="Path to exps folder")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Path for aggregated outputs")
    parser.add_argument("--scratch-dir", type=Path, default=None, help="mockdock scratch dir")
    
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
    print("Generating Figure 1 (Generation Metrics)...")
    plot_figure_1_generation(full_df, fig_dir)
    
    print("Generating Figure 2 (Optimization Metrics)...")
    plot_figure_2_optimization(full_df, fig_dir)
    
    print("Generating Figure 3 (Quality Metrics)...")
    plot_figure_3_quality(full_df, fig_dir)
    
    print("Generating Figure 4 (Trajectories)...")
    plot_figure_4_trajectory(args.exps_dir, fig_dir)
    
    print("Writing Table 1 (Macro Summary)...")
    write_table_1_macro_summary(macro_df, args.output_dir)
    
    print(f"\nDone! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
