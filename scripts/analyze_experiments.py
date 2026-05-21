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


MODEL_RENAME_MAP = {
    "acegen-a2c": "A2C",
    "acegen-ahc": "AHC",
    "acegen-ppo": "PPO",
    "acegen-ppod": "PPOD",
    "acegen-reinforce": "REINFORCE",
    "acegen-reinvent": "REINVENT",
    "genmol": "GenMol",
    "libinvent": "Libinvent",
}


def setup_plotting():
    """Configure seaborn/matplotlib for publication-quality output."""
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})
    sns.set_palette("colorblind")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Helvetica",
        "Avant Garde",
        "sans-serif",
    ]
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
            if not run_dir.is_dir() or not (
                run_dir.name.startswith("run_") or run_dir.name.startswith("202")
            ):
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


from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single(args_tuple):
    model, seed, target, csv_path, scratch_dir, force = args_tuple
    try:
        from mockdock import MDEvaluator
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from mockdock import MDEvaluator

    try:
        # Check if metrics already computed to save time
        metrics_json = csv_path.parent / "eval_metrics.json"
        if metrics_json.exists() and not force:
            with open(metrics_json) as f:
                metrics = json.load(f)
        else:
            evaluator = MDEvaluator(target, scratch_dir=scratch_dir)
            metrics = evaluator.compute_metrics(csv_path)

        # Optional runtime metadata from oracle-side metrics.json
        runtime_path = csv_path.parent / "metrics.json"
        runtime_metrics = _read_runtime_metrics(runtime_path)

        mapped_model = MODEL_RENAME_MAP.get(model.lower(), model)
        # Flat row data
        row = {
            "model": mapped_model,
            "seed": seed,
            "target": target,
        }
        # Add metrics (minus descriptions)
        for k, v in metrics.items():
            if k != "descriptions" and not isinstance(v, dict):
                row[k] = v
        row.update(runtime_metrics)
        return row
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return None


def process_all(
    results_list: list[tuple[str, str, str, Path]], scratch_dir: Path = None, force: bool = False
) -> pl.DataFrame:
    """
    Process each results.csv through MDEvaluator.
    Returns a master DataFrame with all metrics.
    """
    all_data = []

    # Prepare arguments for multiprocessing
    tasks = [(model, seed, target, csv_path, scratch_dir, force) for model, seed, target, csv_path in results_list]

    import os
    max_workers = min(32, os.cpu_count() or 4)
    print(f"Processing {len(tasks)} runs in parallel using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, task): task for task in tasks}

        for future in as_completed(futures):
            model, seed, target, csv_path, _ = futures[future][:5]
            try:
                row = future.result()
                if row is not None:
                    print(f"Successfully evaluated {model} | {target} | {seed}")
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
    plot_types: dict[str, str] = None,
):
    """Shared panel plot helper: one panel per metric with benchmark-wise comparisons."""
    setup_plotting()
    available = [
        (metric, label) for metric, label in metrics_map.items() if metric in full_df.columns
    ]
    if not available:
        return

    pdf = full_df.select(["model", "target"] + [m for m, _ in available]).to_pandas()
    targets = sorted(pdf["target"].unique())
    n_panels = len(available)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    # Consistent ordering for models in the grouped bars
    preferred_order = ["A2C", "AHC", "PPO", "PPOD", "REINFORCE", "REINVENT", "Libinvent", "GenMol"]
    unique_models = list(pdf["model"].unique())
    hue_order = [m for m in preferred_order if m in unique_models] + [
        m for m in unique_models if m not in preferred_order
    ]

    plot_types = plot_types or {}

    for idx, (metric, label) in enumerate(available):
        ax = axes_flat[idx]
        ptype = plot_types.get(metric, "bar")
        if ptype == "box":
            sns.boxplot(
                data=pdf,
                x="target",
                y=metric,
                hue="model",
                hue_order=hue_order,
                order=targets,
                palette=sns.color_palette("colorblind", len(hue_order)),
                linewidth=1.0,
                fliersize=3.0,
                ax=ax,
            )
        else:
            sns.barplot(
                data=pdf,
                x="target",
                y=metric,
                hue="model",
                hue_order=hue_order,
                order=targets,
                estimator=np.mean,
                errorbar=("ci", 95),
                capsize=0.05,
                err_kws={"linewidth": 1.2},
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                ax=ax,
            )
        ax.set_title(label, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Benchmark Target", fontsize=12, labelpad=8)
        ax.set_ylabel("Metric Value", fontsize=12, labelpad=8)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="x", rotation=30)
        
        # Gridlines on Y-axis only
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, top=True, right=True)
        
        # Remove individual subplots' legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(8, len(labels)),
            frameon=False,
            fontsize=13,
        )
    fig.suptitle(figure_title, y=1.03, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_1_generation(full_df: pl.DataFrame, output_dir: Path):
    """Figure 1: Intrinsic/extrinsic generation quality across benchmarks."""
    metrics_map = {
        "validity": "Validity",
        "uniqueness": "Uniqueness",
        "fragment_incorporation": "Fragment 2D",
        "novelty": "Novelty",
    }
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig1_generation_metrics",
        figure_title="Generation Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=4,
    )


def plot_figure_2_optimization(full_df: pl.DataFrame, output_dir: Path):
    """Figure 2: Optimization endpoints across benchmarks."""
    metrics_map = {
        "avg_top_10": "Avg Top-10 Score",
        "avg_top_100": "Avg Top-100 Score",
        "auc_top_10": "AUC Top-10",
        "oracle_efficiency_80": "Oracle Efficiency @80%",
    }
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig2_optimization_metrics",
        figure_title="Optimization Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=4,
    )


def plot_figure_3_quality(results_list: list[tuple[str, str, str, Path]], full_df: pl.DataFrame, output_dir: Path):
    """Figure 3: Medicinal chemistry quality, using box plots for continuous QED/SA and a bar plot for MedChem rate."""
    setup_plotting()
    
    # Gather/Load molecule-level data for QED/SA
    all_mols_data = []
    
    import sys
    from rdkit import Chem
    from rdkit.Chem import QED
    
    try:
        from mockdock.evaluator import MDEvaluator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from mockdock.evaluator import MDEvaluator
        
    print("Gathering molecule-level QED and SA for Figure 3...")
    for model, seed, target, csv_path in results_list:
        cache_path = csv_path.parent / "molecule_metrics_cache.csv"
        mapped_model = MODEL_RENAME_MAP.get(model.lower(), model)
        
        if cache_path.exists():
            try:
                run_df = pl.read_csv(cache_path)
                for row_data in run_df.iter_rows(named=True):
                    all_mols_data.append({
                        "model": mapped_model,
                        "target": target,
                        "seed": seed,
                        "qed": row_data["qed"],
                        "sa": row_data["sa"],
                        "is_novel": row_data["is_novel"],
                    })
                continue
            except Exception as e:
                print(f"  Error reading cache {cache_path}: {e}. Recomputing...")
                
        try:
            evaluator = MDEvaluator(target)
            df = pl.read_csv(csv_path)
            smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
            raw_smiles = df[smiles_col].to_list()
            
            valid_mols = []
            seen = set()
            for s in raw_smiles:
                mol = Chem.MolFromSmiles(str(s))
                if mol is None:
                    continue
                canonical_s = Chem.MolToSmiles(mol)
                if canonical_s not in seen:
                    seen.add(canonical_s)
                    valid_mols.append((canonical_s, mol))
            
            ref_smiles_df = evaluator._loader.get_initial_compounds()
            ref_smiles = set()
            if not ref_smiles_df.is_empty():
                col = "canonical_smiles" if "canonical_smiles" in ref_smiles_df.columns else "smiles"
                ref_smiles = set(ref_smiles_df[col].to_list())
            ref_smiles_canonical = evaluator._canonicalize_smiles_set(ref_smiles)
            
            cache_rows = []
            for smiles, mol in valid_mols:
                is_novel = smiles not in ref_smiles_canonical
                qed_val = float(QED.qed(mol))
                sa_val = float(evaluator._sa_score(mol))
                
                cache_rows.append({
                    "qed": qed_val,
                    "sa": sa_val,
                    "is_novel": is_novel,
                })
                all_mols_data.append({
                    "model": mapped_model,
                    "target": target,
                    "seed": seed,
                    "qed": qed_val,
                    "sa": sa_val,
                    "is_novel": is_novel,
                })
                
            if cache_rows:
                pl.DataFrame(cache_rows).write_csv(cache_path)
                
        except Exception as e:
            print(f"  Error processing molecules for {csv_path}: {e}")
            
    if not all_mols_data:
        print("No molecule-level data found for Figure 3.")
        return
        
    mol_df = pl.DataFrame(all_mols_data)
    
    # Filter for novel compounds
    novel_mol_df = mol_df.filter(pl.col("is_novel")).to_pandas()
    if novel_mol_df.empty:
        novel_mol_df = mol_df.to_pandas()
        qed_label = "QED (All Unique)"
        sa_label = "SA Score (All Unique)"
    else:
        qed_label = "QED (Novel Only)"
        sa_label = "SA Score (Novel Only)"
        
    medchem_df = full_df.select(["model", "target", "fraction_medchem_pass"]).to_pandas()
    
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8))
    
    preferred_order = ["A2C", "AHC", "PPO", "PPOD", "REINFORCE", "REINVENT", "Libinvent", "GenMol"]
    unique_models = list(novel_mol_df["model"].unique())
    hue_order = [m for m in preferred_order if m in unique_models] + [
        m for m in unique_models if m not in preferred_order
    ]
    targets = sorted(novel_mol_df["target"].unique())
    
    # Panel A: QED (Box plot)
    sns.boxplot(
        data=novel_mol_df,
        x="target",
        y="qed",
        hue="model",
        hue_order=hue_order,
        order=targets,
        palette=sns.color_palette("colorblind", len(hue_order)),
        linewidth=1.0,
        fliersize=2.0,
        ax=axes[0],
    )
    axes[0].set_title(qed_label, fontsize=15, fontweight="bold", pad=12)
    axes[0].set_xlabel("Benchmark Target", fontsize=12, labelpad=8)
    axes[0].set_ylabel("QED Score", fontsize=12, labelpad=8)
    
    # Panel B: SA Score (Box plot)
    sns.boxplot(
        data=novel_mol_df,
        x="target",
        y="sa",
        hue="model",
        hue_order=hue_order,
        order=targets,
        palette=sns.color_palette("colorblind", len(hue_order)),
        linewidth=1.0,
        fliersize=2.0,
        ax=axes[1],
    )
    axes[1].set_title(sa_label, fontsize=15, fontweight="bold", pad=12)
    axes[1].set_xlabel("Benchmark Target", fontsize=12, labelpad=8)
    axes[1].set_ylabel("SA Score (lower is better)", fontsize=12, labelpad=8)
    
    # Panel C: Fraction MedChem Pass (Bar plot)
    sns.barplot(
        data=medchem_df,
        x="target",
        y="fraction_medchem_pass",
        hue="model",
        hue_order=hue_order,
        order=targets,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.05,
        err_kws={"linewidth": 1.2},
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        ax=axes[2],
    )
    axes[2].set_title("Fraction Passing MedChem Filters", fontsize=15, fontweight="bold", pad=12)
    axes[2].set_xlabel("Benchmark Target", fontsize=12, labelpad=8)
    axes[2].set_ylabel("Pass Rate", fontsize=12, labelpad=8)
    
    for ax in axes:
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, top=True, right=True)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(8, len(labels)),
        frameon=False,
        fontsize=13,
    )
    
    fig.suptitle("Chemical Quality Metrics Across Benchmarks", y=1.05, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path_stem = output_dir / "fig3_quality_metrics"
    fig.savefig(output_path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Figure 3 with true molecule-level box plots successfully!")



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
        plt.figure(figsize=(10, 6))
        model_results: dict[str, list[Path]] = {}
        for model_dir in exps_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            mapped_model = MODEL_RENAME_MAP.get(model_name.lower(), model_name)
            paths = []
            for run_dir in model_dir.iterdir():
                csv = run_dir / target / "results.csv"
                if csv.exists():
                    paths.append(csv)
            if paths:
                model_results[mapped_model] = paths
        if not model_results:
            continue

        # Sort model results by preferred order
        preferred_order = ["A2C", "AHC", "PPO", "PPOD", "REINFORCE", "REINVENT", "Libinvent", "GenMol"]
        sorted_model_results = {}
        for m in preferred_order:
            if m in model_results:
                sorted_model_results[m] = model_results[m]
        for m in model_results:
            if m not in sorted_model_results:
                sorted_model_results[m] = model_results[m]

        palette = sns.color_palette("colorblind", len(sorted_model_results))
        for i, (model, csv_paths) in enumerate(sorted_model_results.items()):
            curves = []
            for csv in csv_paths:
                df = pl.read_csv(csv)
                score_col = "reward_score" if "reward_score" in df.columns else "norm_score"
                scores = df[score_col].to_list()
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
            plt.fill_between(x, q25, q75, alpha=0.15, color=palette[i])

        plt.xlabel("Cumulative Oracle Calls", fontsize=12, labelpad=8)
        plt.ylabel(f"Running Avg Top-{k} Reward Score", fontsize=12, labelpad=8)
        plt.title(f"Optimization Trajectory: {target} (Median with IQR, k={k})", fontsize=14, fontweight="bold", pad=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig4_trajectory_{target}.svg", bbox_inches="tight")
        plt.savefig(output_dir / f"fig4_trajectory_{target}.png", dpi=300, bbox_inches="tight")
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Path for aggregated outputs"
    )
    parser.add_argument("--scratch-dir", type=Path, default=None, help="mockdock scratch dir")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all metrics")

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
    full_df = process_all(results_list, scratch_dir=args.scratch_dir, force=args.force)
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
    plot_figure_3_quality(results_list, full_df, fig_dir)

    print("Generating Figure 4 (Trajectories)...")
    plot_figure_4_trajectory(args.exps_dir, fig_dir)

    print("Writing Table 1 (Macro Summary)...")
    write_table_1_macro_summary(macro_df, args.output_dir)

    print(f"\nDone! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
