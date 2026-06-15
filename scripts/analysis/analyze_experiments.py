#!/usr/bin/env python3
# scripts/analysis/analyze_experiments.py
"""
Aggregates mockdock experiment results across models, targets, and seeds.
Generates master CSVs and publication-quality figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.experiment_utils import (
    EXPENSIVE_SCORE_COLUMNS,
    MODEL_PLOT_ORDER,
    MODEL_RENAME_MAP,
    REFERENCE_SET_CACHE_DIRNAME,
    REFERENCE_SET_CACHE_FILENAME,
    REFERENCE_SET_LABEL,
    ensure_src_on_path,
)

try:
    from mockdock import MDEvaluator
except ImportError:
    ensure_src_on_path()
    from mockdock import MDEvaluator


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
        ensure_src_on_path()
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


def get_model_palette(models: list[str]) -> dict[str, str | tuple]:
    """Get a consistent color palette mapping for the models, where the reference set is grey."""
    base_palette = sns.color_palette("colorblind")
    palette_map = {}
    
    # Filter out Reference Set from standard models ordering to assign colors
    actual_models_order = [m for m in MODEL_PLOT_ORDER if m != REFERENCE_SET_LABEL]
    
    color_idx = 0
    for m in actual_models_order:
        if m == "GenMol":
            palette_map[m] = base_palette[9]
        else:
            palette_map[m] = base_palette[color_idx % len(base_palette)]
        color_idx += 1
        
    for m in models:
        if m != REFERENCE_SET_LABEL and m not in palette_map:
            palette_map[m] = base_palette[color_idx % len(base_palette)]
            color_idx += 1
            
    palette_map[REFERENCE_SET_LABEL] = "#808080"
    return palette_map


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

    unique_models = list(pdf["model"].unique())
    hue_order = [m for m in MODEL_PLOT_ORDER if m in unique_models] + [
        m for m in unique_models if m not in MODEL_PLOT_ORDER
    ]
    palette = get_model_palette(unique_models)

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
                palette=palette,
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
                palette=palette,
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
        "effective_hit_rate": "Effective Yield Rate",
        "effective_medchem_pass": "MedChem Pass Rate",
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
        "avg_top_10": "Avg Top-10 Score",
        "avg_top_100": "Avg Top-100 Score",
        "auc_top_10": "AUC Top-10",
    }
    _plot_metric_panels(
        full_df=full_df,
        metrics_map=metrics_map,
        output_path_stem=output_dir / "fig2_optimization_metrics",
        figure_title="Optimization Metrics Across Benchmarks (mean +/- 95% CI)",
        ncols=3,
    )


def _reference_set_cache_path(reference_cache_dir: Path, target: str) -> Path:
    return reference_cache_dir / target / REFERENCE_SET_CACHE_FILENAME


def _join_quality_score_files(cache_df: pl.DataFrame, cache_dir: Path) -> pl.DataFrame:
    """Join optional expensive per-molecule score files beside a molecule cache.

    Expensive scores live in scores_*.csv files. If an older cache still has score
    columns, drop them before joining so stale cache values cannot shadow reruns.
    """
    legacy_score_cols = [col for col in EXPENSIVE_SCORE_COLUMNS if col in cache_df.columns]
    if legacy_score_cols:
        cache_df = cache_df.drop(legacy_score_cols)

    score_files = [
        (cache_dir / "scores_molskill.csv", {"molskill_score"}),
        (cache_dir / "scores_stoplight.csv", {"stoplight_score"}),
        (
            cache_dir / "scores_aizynthfinder.csv",
            {"aizynthfinder_score", "aizynthfinder_state_score"},
        ),
    ]
    for score_path, score_cols in score_files:
        if not score_path.exists():
            continue
        existing_cols = set(cache_df.columns)
        try:
            score_df = pl.read_csv(score_path)
            cols_to_add = ["smiles"] + [
                col for col in score_cols if col in score_df.columns and col not in existing_cols
            ]
            if len(cols_to_add) > 1:
                cache_df = cache_df.join(score_df.select(cols_to_add), on="smiles", how="left")
        except Exception as e:
            print(f"  Error joining {score_path.name}: {e}")
    return cache_df


def _build_reference_set_cache(
    target: str,
    cache_path: Path,
    chem_module,
    qed_module,
    evaluator_cls,
    force: bool,
) -> pl.DataFrame:
    """Compute and cache baseline quality metrics for one bundled reference set."""
    required_cols = {"smiles", "qed", "sa", "is_novel", "has_fragment", "passes_medchem"}
    if cache_path.exists() and not force:
        try:
            cached_df = pl.read_csv(cache_path)
            if required_cols.issubset(set(cached_df.columns)):
                return cached_df
            print(f"  Reference cache {cache_path} missing required columns, recomputing...")
        except Exception as e:
            print(f"  Error reading reference cache {cache_path}: {e}. Recomputing...")

    evaluator = evaluator_cls(target)
    reference_df, _, _ = evaluator._loader.get_full_data_and_threshold()
    if reference_df.is_empty():
        return pl.DataFrame()

    smiles_col = (
        "canonical_smiles"
        if "canonical_smiles" in reference_df.columns
        else "smiles"
        if "smiles" in reference_df.columns
        else None
    )
    if smiles_col is None:
        print(f"  Reference data for {target} has no SMILES column.")
        return pl.DataFrame()

    seen = set()
    rows = []
    for row_data in reference_df.iter_rows(named=True):
        mol = chem_module.MolFromSmiles(str(row_data[smiles_col]))
        if mol is None:
            continue
        smiles = chem_module.MolToSmiles(mol)
        if smiles in seen:
            continue
        seen.add(smiles)
        rows.append(
            {
                "smiles": smiles,
                "qed": float(qed_module.qed(mol)),
                "sa": float(evaluator._sa_score(mol)),
                # Reference-set compounds define the baseline; keep all rows in Figure 3.
                "is_novel": True,
                "has_fragment": True,
                "passes_medchem": True,
            }
        )

    cache_df = pl.DataFrame(rows)
    if not cache_df.is_empty():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.write_csv(cache_path)
        print(f"  Saved reference cache for {target} to {cache_path}")
    return cache_df


def _append_reference_quality_rows(
    all_mols_data: list[dict],
    targets: list[str],
    reference_cache_dir: Path,
    chem_module,
    qed_module,
    evaluator_cls,
    force: bool = False,
):
    """Add cached benchmark reference-set molecules to Figure 3's molecule-level rows."""
    for target in targets:
        try:
            cache_path = _reference_set_cache_path(reference_cache_dir, target)
            reference_df = _build_reference_set_cache(
                target=target,
                cache_path=cache_path,
                chem_module=chem_module,
                qed_module=qed_module,
                evaluator_cls=evaluator_cls,
                force=force,
            )
            if reference_df.is_empty():
                print(f"  No reference-set data found for {target}.")
                continue

            reference_df = _join_quality_score_files(reference_df, cache_path.parent)
            added = 0
            for row_data in reference_df.iter_rows(named=True):
                all_mols_data.append(
                    {
                        "model": REFERENCE_SET_LABEL,
                        "target": target,
                        "seed": "reference",
                        "smiles": row_data["smiles"],
                        "qed": row_data["qed"],
                        "sa": row_data["sa"],
                        "is_novel": True,
                        "has_fragment": True,
                        "passes_medchem": True,
                        "molskill_score": row_data.get("molskill_score", None),
                        "stoplight_score": row_data.get("stoplight_score", None),
                        "aizynthfinder_score": row_data.get("aizynthfinder_score", None),
                        "aizynthfinder_state_score": row_data.get(
                            "aizynthfinder_state_score", None
                        ),
                    }
                )
                added += 1
            print(f"  Added {added} reference-set molecules for {target}.")
        except Exception as e:
            print(f"  Error processing reference-set molecules for {target}: {e}")


def plot_figure_3_quality(
    results_list: list[tuple[str, str, str, Path]],
    full_df: pl.DataFrame,
    output_dir: Path,
    reference_cache_dir: Path,
    force: bool = False,
):
    """Figure 3: Chemical quality metrics on the Effective Yield Rate compound set.

    The Effective Yield Rate set consists of molecules that are valid, unique,
    contain the 2D fragment substructure, and are novel against the model-visible set.
    QED, SA Score, and MedChem pass rate are all computed on this consistent set.
    Optional expensive scores are loaded from sibling scores_*.csv files.
    """
    setup_plotting()

    # Gather/Load molecule-level data
    all_mols_data = []

    from rdkit import Chem
    from rdkit.Chem import QED

    try:
        from mockdock.evaluator import MDEvaluator
        from mockdock.filters import MDFilters
        from mockdock.utils import get_robust_match
    except ImportError:
        ensure_src_on_path()
        from mockdock.evaluator import MDEvaluator
        from mockdock.filters import MDFilters
        from mockdock.utils import get_robust_match

    filters = MDFilters(active_rulesets=["PAINS", "BMS"])

    print("Gathering molecule-level data for Figure 3 (Effective Yield Rate set)...")
    for model, seed, target, csv_path in results_list:
        cache_path = csv_path.parent / "molecule_metrics_cache.csv"
        mapped_model = MODEL_RENAME_MAP.get(model.lower(), model)

        # Check if cache exists with all required columns
        use_cache = False
        if cache_path.exists():
            try:
                run_df = pl.read_csv(cache_path)
                required_cols = {"smiles", "qed", "sa", "is_novel", "has_fragment", "passes_medchem"}
                if required_cols.issubset(set(run_df.columns)):
                    run_df = _join_quality_score_files(run_df, csv_path.parent)
                    use_cache = True
                    for row_data in run_df.iter_rows(named=True):
                        all_mols_data.append({
                            "model": mapped_model,
                            "target": target,
                            "seed": seed,
                            "smiles": row_data["smiles"],
                            "qed": row_data["qed"],
                            "sa": row_data["sa"],
                            "is_novel": row_data["is_novel"],
                            "has_fragment": row_data["has_fragment"],
                            "passes_medchem": row_data["passes_medchem"],
                            "molskill_score": row_data.get("molskill_score", None),
                            "stoplight_score": row_data.get("stoplight_score", None),
                            "aizynthfinder_score": row_data.get("aizynthfinder_score", None),
                            "aizynthfinder_state_score": row_data.get("aizynthfinder_state_score", None),
                        })
                else:
                    print(f"  Cache {cache_path} missing required columns, recomputing...")
            except Exception as e:
                print(f"  Error reading cache {cache_path}: {e}. Recomputing...")

        if use_cache:
            continue

        try:
            evaluator = MDEvaluator(target)
            fragment_smiles = evaluator._loader.fragment_smiles
            frag_q = Chem.MolFromSmiles(fragment_smiles) if fragment_smiles else None
            if frag_q is None and fragment_smiles:
                frag_q = Chem.MolFromSmarts(fragment_smiles)

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
                has_fragment = bool(frag_q is not None and get_robust_match(mol, frag_q))
                qed_val = float(QED.qed(mol))
                sa_val = float(evaluator._sa_score(mol))
                filter_result = filters.evaluate(mol)
                passes_medchem = filter_result["pass"]

                cache_rows.append({
                    "smiles": smiles,
                    "qed": qed_val,
                    "sa": sa_val,
                    "is_novel": is_novel,
                    "has_fragment": has_fragment,
                    "passes_medchem": passes_medchem,
                })
                all_mols_data.append({
                    "model": mapped_model,
                    "target": target,
                    "seed": seed,
                    "smiles": smiles,
                    "qed": qed_val,
                    "sa": sa_val,
                    "is_novel": is_novel,
                    "has_fragment": has_fragment,
                    "passes_medchem": passes_medchem,
                    "molskill_score": None,
                    "stoplight_score": None,
                    "aizynthfinder_score": None,
                    "aizynthfinder_state_score": None,
                })

            if cache_rows:
                pl.DataFrame(cache_rows).write_csv(cache_path)

        except Exception as e:
            print(f"  Error processing molecules for {csv_path}: {e}")

    print("Gathering reference-set molecule-level data for Figure 3...")
    reference_targets = sorted({target for _, _, target, _ in results_list})
    _append_reference_quality_rows(
        all_mols_data=all_mols_data,
        targets=reference_targets,
        reference_cache_dir=reference_cache_dir,
        chem_module=Chem,
        qed_module=QED,
        evaluator_cls=MDEvaluator,
        force=force,
    )

    if not all_mols_data:
        print("No molecule-level data found for Figure 3.")
        return

    mol_df = pl.DataFrame(all_mols_data)

    # Ensure all 5 score columns exist in the DataFrame, filling with None if missing
    for col in ["qed", "sa", "molskill_score", "stoplight_score", "aizynthfinder_state_score"]:
        if col not in mol_df.columns:
            mol_df = mol_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Filter for Effective Yield Rate compound set: valid + unique + novel + fragment 2D
    effective_df = mol_df.filter(
        pl.col("is_novel") & pl.col("has_fragment")
    ).to_pandas()

    if effective_df.empty:
        print("Warning: No molecules in Effective Yield Rate set. Using all unique molecules.")
        effective_df = mol_df.to_pandas()
        set_label = " (All Unique)"
    else:
        set_label = ""

    # ── Panel layout ─────────────────────────────────────────────────────
    # Active panels: dynamically render subplots that have valid data
    plot_molskill = "molskill_score" in effective_df.columns and effective_df["molskill_score"].notna().any()
    plot_stoplight = "stoplight_score" in effective_df.columns and effective_df["stoplight_score"].notna().any()
    plot_aizynth = "aizynthfinder_state_score" in effective_df.columns and effective_df["aizynthfinder_state_score"].notna().any()

    active_panels = ["qed", "sa"]
    if plot_molskill:
        active_panels.append("molskill_score")
    if plot_stoplight:
        active_panels.append("stoplight_score")
    if plot_aizynth:
        active_panels.append("aizynthfinder_state_score")

    n_panels = len(active_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.8), squeeze=False)
    axes_flat = axes.flatten()

    unique_models = list(effective_df["model"].unique())
    hue_order = [m for m in MODEL_PLOT_ORDER if m in unique_models] + [
        m for m in unique_models if m not in MODEL_PLOT_ORDER
    ]
    targets = sorted(effective_df["target"].unique())
    palette = get_model_palette(unique_models)

    for idx, col_name in enumerate(active_panels):
        ax = axes_flat[idx]
        sns.boxplot(
            data=effective_df,
            x="target",
            y=col_name,
            hue="model",
            hue_order=hue_order,
            order=targets,
            palette=palette,
            linewidth=1.0,
            fliersize=2.0,
            ax=ax,
        )
        if col_name == "qed":
            ax.set_title(f"QED (\u2191){set_label}", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel("QED Score", fontsize=12, labelpad=8)
        elif col_name == "sa":
            ax.set_title(f"SA Score (\u2193){set_label}", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel("SA Score", fontsize=12, labelpad=8)
        elif col_name == "molskill_score":
            ax.set_title(f"MolSkill Score (\u2193){set_label}", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel("MolSkill Score", fontsize=12, labelpad=8)
        elif col_name == "stoplight_score":
            ax.set_title(f"Stoplight Score (\u2193){set_label}", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel("Stoplight Score", fontsize=12, labelpad=8)
        elif col_name == "aizynthfinder_state_score":
            ax.set_title(f"AIZynthFinder State Score (\u2191){set_label}", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel("State Score", fontsize=12, labelpad=8)

        ax.set_xlabel("Benchmark Target", fontsize=12, labelpad=8)

    for ax in axes_flat:
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, top=True, right=True)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
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
    print("Generated Figure 3 with Effective Yield Rate compound set!")



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

        palette = get_model_palette(list(sorted_model_results.keys()))
        for model, csv_paths in sorted_model_results.items():
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
            
            n_seeds = aligned.shape[0]
            mean_curve = np.mean(aligned, axis=0)
            if n_seeds > 1:
                sem = np.std(aligned, axis=0, ddof=1) / np.sqrt(n_seeds)
                ci = 1.96 * sem
            else:
                ci = np.zeros_like(mean_curve)
            lower = mean_curve - ci
            upper = mean_curve + ci
            
            x = np.arange(1, min_len + 1)
            color = palette[model]
            plt.plot(x, mean_curve, label=model, lw=2.2, color=color)
            plt.fill_between(x, lower, upper, alpha=0.15, color=color)

        plt.xlabel("Cumulative Oracle Calls", fontsize=12, labelpad=8)
        plt.ylabel(f"Running Top-{k} Mean Reward Score", fontsize=12, labelpad=8)
        plt.title(f"Optimization Trajectory: {target}", fontsize=14, fontweight="bold", pad=12)
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
        "effective_hit_rate_mean",
        "effective_medchem_pass_mean",
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
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=None,
        help=(
            "Shared cache directory for reference-set molecule metrics and optional "
            "MolSkill/STOPLIGHT/AIZynthFinder scores"
        ),
    )
    parser.add_argument("--scratch-dir", type=Path, default=None, help="mockdock scratch dir")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all metrics")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    reference_cache_dir = args.reference_cache_dir or (
        args.exps_dir.parent / REFERENCE_SET_CACHE_DIRNAME
    )

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
    plot_figure_3_quality(
        results_list,
        full_df,
        fig_dir,
        reference_cache_dir=reference_cache_dir,
        force=args.force,
    )

    print("Generating Figure 4 (Trajectories)...")
    plot_figure_4_trajectory(args.exps_dir, fig_dir)

    print("Writing Table 1 (Macro Summary)...")
    write_table_1_macro_summary(macro_df, args.output_dir)

    print(f"\nDone! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
