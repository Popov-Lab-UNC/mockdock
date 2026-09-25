#!/usr/bin/env python3
"""Calculate statistical distribution tests comparing generated vs reference molecules.

Accounts for low or zero Effective Yield Rate (EYR) sample sizes across runs and targets.

Metrics evaluated:
- QED (higher is better)
- SA Score (lower is better)
- MolSkill Score (lower is better)
- Stoplight Score (lower is better)
- AIZynthFinder State Score (higher is better)

Statistical tests and effect sizes computed:
1. Wasserstein Distance (W1 / Earth Mover's Distance)
2. Two-sample Kolmogorov-Smirnov test (D_KS statistic and p-value)
3. Mann-Whitney U test (p-value) and Cliff's Delta (delta effect size)
4. Central tendencies: Median, IQR, Mean, Std for both reference and model

Low EYR Handling:
- Uses a minimum sample threshold (default N_min = 10) for individual run tests.
- Flags runs with N_EYR < N_min as insufficient_sample and excludes them from seed averages.
- Reports n_valid_seeds (e.g. 5/5, 2/5, 0/5) and average N_EYR for full transparency.
- Evaluates pooled molecules across all 5 seeds for robust composite assessment.
- Computes macro-averages over Tractable targets (5 targets where all models have sufficient yield)
  as well as All 7 targets.

Outputs:
- chemical_quality_stats_runs.csv (run-by-run statistics and sample sizes)
- chemical_quality_stats_target_summary.csv (mean +/- std across qualifying seeds per target)
- chemical_quality_stats_macro_summary.csv (macro-averaged across targets)
- chemical_quality_stats_pooled.csv (pooled across all seeds)
- table_quality_vs_reference.tex (publication LaTeX table)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.experiment_utils import (
    MODEL_PLOT_ORDER,
    MODEL_RENAME_MAP,
    REFERENCE_SET_CACHE_FILENAME,
    REFERENCE_SET_LABEL,
    effective_yield_filter,
)

METRIC_CONFIGS = [
    {
        "name": "qed",
        "label": "QED",
        "higher_is_better": True,
        "format": ".3f",
    },
    {
        "name": "sa",
        "label": "SA Score",
        "higher_is_better": False,
        "format": ".3f",
    },
    {
        "name": "molskill_score",
        "label": "MolSkill",
        "higher_is_better": False,
        "format": ".2f",
    },
    {
        "name": "stoplight_score",
        "label": "STOPLIGHT",
        "higher_is_better": False,
        "format": ".3f",
    },
    {
        "name": "aizynthfinder_state_score",
        "label": "AIZynthFinder",
        "higher_is_better": True,
        "format": ".3f",
    },
]

TRACTABLE_TARGETS = ["CHK1", "DPP4", "ITK", "PEPCK", "TTK"]
LOW_YIELD_TARGETS = ["PptT", "VEGFR2"]


def join_expensive_scores(cache_df: pl.DataFrame, directory: Path) -> pl.DataFrame:
    """Join scores_molskill.csv, scores_stoplight.csv, scores_aizynthfinder.csv."""
    existing_cols = set(cache_df.columns)
    score_files = [
        (directory / "scores_molskill.csv", ["smiles", "molskill_score"]),
        (directory / "scores_stoplight.csv", ["smiles", "stoplight_score"]),
        (directory / "scores_aizynthfinder.csv", ["smiles", "aizynthfinder_state_score"]),
    ]
    for score_path, cols in score_files:
        if not score_path.exists():
            continue
        try:
            score_df = pl.read_csv(score_path)
            cols_to_add = [c for c in cols if c in score_df.columns and (c == "smiles" or c not in existing_cols)]
            if len(cols_to_add) > 1:
                cache_df = cache_df.join(score_df.select(cols_to_add), on="smiles", how="left")
                existing_cols.update(cols_to_add[1:])
        except Exception as e:
            print(f"  Warning: could not join {score_path.name} from {directory}: {e}")
    return cache_df


def load_reference_sets(ref_dir: Path) -> dict[str, pl.DataFrame]:
    """Load reference set molecules for each target with all scores joined."""
    ref_data = {}
    if not ref_dir.exists():
        print(f"Error: reference directory {ref_dir} does not exist.")
        return ref_data

    for target_path in sorted(ref_dir.iterdir()):
        if not target_path.is_dir():
            continue
        target = target_path.name
        cache_csv = target_path / REFERENCE_SET_CACHE_FILENAME
        if not cache_csv.exists():
            continue
        df = pl.read_csv(cache_csv)
        df = join_expensive_scores(df, target_path)
        ref_data[target] = df
        print(f"Loaded reference set for {target}: {len(df)} compounds")
    return ref_data


def compute_distribution_stats(
    gen_vals: np.ndarray,
    ref_vals: np.ndarray,
    higher_is_better: bool,
    min_samples: int = 10,
) -> dict[str, Any]:
    """Compute comprehensive statistical metrics between generated and reference values.

    If gen_clean has fewer than min_samples, returns stats with insufficient_sample=True
    to avoid reporting unrepresentative metrics.
    """
    gen_clean = gen_vals[~np.isnan(gen_vals)]
    ref_clean = ref_vals[~np.isnan(ref_vals)]

    n_gen = len(gen_clean)
    n_ref = len(ref_clean)

    base_dict = {
        "n_gen": n_gen,
        "n_ref": n_ref,
        "insufficient_sample": n_gen < min_samples,
    }

    if n_ref == 0:
        return {
            **base_dict,
            "ref_median": np.nan, "ref_iqr": np.nan, "ref_mean": np.nan, "ref_std": np.nan,
            "gen_median": np.nan, "gen_iqr": np.nan, "gen_mean": np.nan, "gen_std": np.nan,
            "median_diff": np.nan, "mean_diff": np.nan,
            "wasserstein_dist": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan,
            "mwu_pvalue": np.nan, "cliffs_delta": np.nan, "superior_effect": np.nan,
        }

    ref_med = float(np.median(ref_clean))
    ref_iqr = float(np.percentile(ref_clean, 75) - np.percentile(ref_clean, 25))
    ref_mean = float(np.mean(ref_clean))
    ref_std = float(np.std(ref_clean, ddof=1)) if n_ref > 1 else 0.0

    if n_gen == 0:
        return {
            **base_dict,
            "ref_median": ref_med, "ref_iqr": ref_iqr, "ref_mean": ref_mean, "ref_std": ref_std,
            "gen_median": np.nan, "gen_iqr": np.nan, "gen_mean": np.nan, "gen_std": np.nan,
            "median_diff": np.nan, "mean_diff": np.nan,
            "wasserstein_dist": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan,
            "mwu_pvalue": np.nan, "cliffs_delta": np.nan, "superior_effect": np.nan,
        }

    gen_med = float(np.median(gen_clean))
    gen_iqr = float(np.percentile(gen_clean, 75) - np.percentile(gen_clean, 25))
    gen_mean = float(np.mean(gen_clean))
    gen_std = float(np.std(gen_clean, ddof=1)) if n_gen > 1 else 0.0
    median_diff = gen_med - ref_med
    mean_diff = gen_mean - ref_mean

    # If below threshold, calculate central tendencies but do not compute distribution tests
    if n_gen < min_samples:
        return {
            **base_dict,
            "ref_median": ref_med, "ref_iqr": ref_iqr, "ref_mean": ref_mean, "ref_std": ref_std,
            "gen_median": gen_med, "gen_iqr": gen_iqr, "gen_mean": gen_mean, "gen_std": gen_std,
            "median_diff": median_diff, "mean_diff": mean_diff,
            "wasserstein_dist": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan,
            "mwu_pvalue": np.nan, "cliffs_delta": np.nan, "superior_effect": np.nan,
        }

    # 1. Wasserstein Distance (W1)
    w1 = float(stats.wasserstein_distance(gen_clean, ref_clean))

    # 2. Two-Sample Kolmogorov-Smirnov Test
    ks_res = stats.ks_2samp(gen_clean, ref_clean)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    # 3. Mann-Whitney U test & Cliff's Delta
    mwu_res = stats.mannwhitneyu(gen_clean, ref_clean, alternative="two-sided")
    u_stat = float(mwu_res.statistic)
    mwu_pval = float(mwu_res.pvalue)

    # Cliff's Delta: delta = (2 * U) / (n_gen * n_ref) - 1
    cliffs_delta = float((2.0 * u_stat) / (n_gen * n_ref) - 1.0)
    superior_effect = cliffs_delta if higher_is_better else -cliffs_delta

    return {
        **base_dict,
        "ref_median": ref_med, "ref_iqr": ref_iqr, "ref_mean": ref_mean, "ref_std": ref_std,
        "gen_median": gen_med, "gen_iqr": gen_iqr, "gen_mean": gen_mean, "gen_std": gen_std,
        "median_diff": median_diff, "mean_diff": mean_diff,
        "wasserstein_dist": w1,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "mwu_pvalue": mwu_pval,
        "cliffs_delta": cliffs_delta,
        "superior_effect": superior_effect,
    }


def analyze_experiments(
    exps_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    min_samples: int = 10,
) -> None:
    """Run chemical quality distribution tests with explicit low-EYR handling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_data = load_reference_sets(ref_dir)
    if not ref_data:
        print("No reference sets found. Aborting.")
        return

    run_records: list[dict[str, Any]] = []
    # Dict to collect pooled generated values: (model, target, metric) -> list of arrays
    pooled_gen_mols: dict[tuple[str, str, str], list[np.ndarray]] = {}

    model_dirs = sorted([d for d in exps_dir.iterdir() if d.is_dir()])
    targets = sorted(ref_data.keys())

    print(f"Processing {len(model_dirs)} models across {len(targets)} targets (N_min = {min_samples})...")

    for model_dir in model_dirs:
        raw_model = model_dir.name
        model_name = MODEL_RENAME_MAP.get(raw_model.lower(), raw_model)

        run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            run_dirs = [model_dir]

        for run_idx, run_dir in enumerate(run_dirs):
            run_name = run_dir.name if run_dir != model_dir else f"run_{run_idx+1}"

            for target in targets:
                target_dir = run_dir / target
                cache_csv = target_dir / "molecule_metrics_cache.csv"
                if not cache_csv.exists():
                    continue

                try:
                    df = pl.read_csv(cache_csv)
                    df = join_expensive_scores(df, target_dir)

                    # Filter for Effective Yield Rate set
                    eyr_df = effective_yield_filter(df)
                    n_eyr_total = len(eyr_df)

                    ref_df = ref_data[target]

                    for mcfg in METRIC_CONFIGS:
                        metric_name = mcfg["name"]
                        higher_is_better = mcfg["higher_is_better"]

                        if metric_name not in eyr_df.columns or metric_name not in ref_df.columns:
                            continue

                        gen_vals = eyr_df[metric_name].drop_nulls().to_numpy()
                        ref_vals = ref_df[metric_name].drop_nulls().to_numpy()

                        # Collect for pooled
                        key = (model_name, target, metric_name)
                        if key not in pooled_gen_mols:
                            pooled_gen_mols[key] = []
                        if len(gen_vals) > 0:
                            pooled_gen_mols[key].append(gen_vals)

                        stats_dict = compute_distribution_stats(
                            gen_vals, ref_vals, higher_is_better, min_samples=min_samples
                        )

                        rec = {
                            "model": model_name,
                            "raw_model": raw_model,
                            "run": run_name,
                            "target": target,
                            "n_eyr_total": n_eyr_total,
                            "metric": metric_name,
                            "metric_label": mcfg["label"],
                            "higher_is_better": higher_is_better,
                            **stats_dict,
                        }
                        run_records.append(rec)

                except Exception as e:
                    print(f"  Error analyzing {model_name} {run_name} {target}: {e}")

    if not run_records:
        print("No records collected.")
        return

    runs_df = pl.DataFrame(run_records)
    runs_csv_path = output_dir / "chemical_quality_stats_runs.csv"
    runs_df.write_csv(runs_csv_path)
    print(f"Saved run-level statistics to {runs_csv_path} ({len(runs_df)} rows)")

    # ── Target-Level Summary (Mean +/- Std across qualifying seeds) ───────────
    summary_stat_cols = [
        "ref_median", "gen_median", "median_diff",
        "ref_mean", "gen_mean", "mean_diff",
        "wasserstein_dist", "ks_statistic", "ks_pvalue",
        "mwu_pvalue", "cliffs_delta", "superior_effect",
    ]

    target_summary_rows = []
    unique_groups = (
        runs_df.select(["model", "target", "metric", "metric_label", "higher_is_better"])
        .unique()
        .iter_rows(named=True)
    )

    for grp in unique_groups:
        sub = runs_df.filter(
            (pl.col("model") == grp["model"])
            & (pl.col("target") == grp["target"])
            & (pl.col("metric") == grp["metric"])
        )
        total_seeds = len(sub)
        n_gen_vals = sub["n_gen"].to_numpy()
        mean_n_gen = float(np.mean(n_gen_vals)) if len(n_gen_vals) > 0 else 0.0

        # Qualifying seeds: runs that met min_samples
        qualifying = sub.filter(~pl.col("insufficient_sample"))
        n_valid_seeds = len(qualifying)

        row = {
            "model": grp["model"],
            "target": grp["target"],
            "metric": grp["metric"],
            "metric_label": grp["metric_label"],
            "higher_is_better": grp["higher_is_better"],
            "total_seeds": total_seeds,
            "n_valid_seeds": n_valid_seeds,
            "mean_n_gen": mean_n_gen,
            "has_sufficient_seeds": n_valid_seeds >= 3,  # At least 3 of 5 seeds
        }

        # Reference values (same across seeds)
        ref_meds = sub["ref_median"].drop_nulls().to_numpy()
        row["ref_median"] = float(ref_meds[0]) if len(ref_meds) > 0 else np.nan

        # Aggregate generated and comparison stats over qualifying seeds
        for col in summary_stat_cols:
            if n_valid_seeds > 0:
                vals = qualifying[col].drop_nulls().to_numpy()
                if len(vals) > 0:
                    row[f"{col}_mean"] = float(np.mean(vals))
                    row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan

        target_summary_rows.append(row)

    target_summary_df = pl.DataFrame(target_summary_rows)
    target_summary_csv = output_dir / "chemical_quality_stats_target_summary.csv"
    target_summary_df.write_csv(target_summary_csv)
    print(f"Saved target-level summary statistics to {target_summary_csv} ({len(target_summary_df)} rows)")

    # ── Pooled Across Seeds Analysis ──────────────────────────────────────────
    pooled_rows = []
    for (model, target, metric_name), val_list in pooled_gen_mols.items():
        if not val_list:
            continue
        gen_pooled = np.concatenate(val_list)
        ref_df = ref_data[target]
        if metric_name not in ref_df.columns:
            continue
        ref_vals = ref_df[metric_name].drop_nulls().to_numpy()
        higher_is_better = next(m["higher_is_better"] for m in METRIC_CONFIGS if m["name"] == metric_name)
        mcfg_label = next(m["label"] for m in METRIC_CONFIGS if m["name"] == metric_name)

        pstats = compute_distribution_stats(gen_pooled, ref_vals, higher_is_better, min_samples=min_samples)
        pooled_rows.append({
            "model": model,
            "target": target,
            "metric": metric_name,
            "metric_label": mcfg_label,
            "higher_is_better": higher_is_better,
            **pstats,
        })

    pooled_df = pl.DataFrame(pooled_rows)
    pooled_csv = output_dir / "chemical_quality_stats_pooled.csv"
    pooled_df.write_csv(pooled_csv)
    print(f"Saved pooled statistics to {pooled_csv} ({len(pooled_df)} rows)")

    # ── Macro-Level Summary (Macro-average across targets per model) ───────────
    macro_rows = []
    scopes = [
        ("Tractable_5_Targets", TRACTABLE_TARGETS),
        ("All_7_Targets_Qualifying_Only", None),
    ]

    for scope_name, target_filter in scopes:
        filtered_summary = target_summary_df
        if target_filter is not None:
            filtered_summary = filtered_summary.filter(pl.col("target").is_in(target_filter))

        for model in sorted(filtered_summary["model"].unique()):
            for mcfg in METRIC_CONFIGS:
                m = mcfg["name"]
                sub = filtered_summary.filter(
                    (pl.col("model") == model) & (pl.col("metric") == m)
                )
                if len(sub) == 0:
                    continue

                # Filter for targets with valid seed data
                valid_targets = sub.filter(pl.col("n_valid_seeds") > 0)
                n_eval_targets = len(valid_targets)

                row = {
                    "scope": scope_name,
                    "model": model,
                    "metric": m,
                    "metric_label": mcfg["label"],
                    "higher_is_better": mcfg["higher_is_better"],
                    "n_total_targets": len(sub),
                    "n_eval_targets": n_eval_targets,
                    "mean_eyr_mols_per_target": float(np.mean(sub["mean_n_gen"].to_numpy())) if len(sub) > 0 else 0.0,
                }

                for col in ["wasserstein_dist", "ks_statistic", "cliffs_delta", "superior_effect", "median_diff"]:
                    mean_col = f"{col}_mean"
                    vals = valid_targets[mean_col].drop_nulls().to_numpy()
                    if len(vals) > 0:
                        row[f"macro_{col}"] = float(np.mean(vals))
                        row[f"macro_{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    else:
                        row[f"macro_{col}"] = np.nan
                        row[f"macro_{col}_std"] = np.nan
                macro_rows.append(row)

    macro_df = pl.DataFrame(macro_rows)
    macro_csv = output_dir / "chemical_quality_stats_macro_summary.csv"
    macro_df.write_csv(macro_csv)
    print(f"Saved macro-averaged statistics to {macro_csv} ({len(macro_df)} rows)")

    # ── Generate LaTeX Table Snippet ──────────────────────────────────────────
    generate_latex_table(
        macro_df=macro_df,
        target_summary_df=target_summary_df,
        latex_path=output_dir / "table_quality_vs_reference.tex",
    )


def generate_latex_table(
    macro_df: pl.DataFrame,
    target_summary_df: pl.DataFrame,
    latex_path: Path,
) -> None:
    """Format macro-level chemical quality metrics into a professional LaTeX table."""
    df_tractable = macro_df.filter(pl.col("scope") == "Tractable_5_Targets").to_pandas()
    if df_tractable.empty:
        df_tractable = macro_df.to_pandas()

    models = [m for m in MODEL_PLOT_ORDER if m != REFERENCE_SET_LABEL and m in df_tractable["model"].unique()]
    models += [m for m in df_tractable["model"].unique() if m not in MODEL_PLOT_ORDER and m != REFERENCE_SET_LABEL]

    metrics = ["qed", "sa", "molskill_score", "stoplight_score", "aizynthfinder_state_score"]
    metric_headers = [
        "\\shortstack{\\textbf{QED} ($\\uparrow$)\\\\$W_1$ | $\\delta$}",
        "\\shortstack{\\textbf{SA Score} ($\\downarrow$)\\\\$W_1$ | $\\delta$}",
        "\\shortstack{\\textbf{MolSkill} ($\\downarrow$)\\\\$W_1$ | $\\delta$}",
        "\\shortstack{\\textbf{STOPLIGHT} ($\\downarrow$)\\\\$W_1$ | $\\delta$}",
        "\\shortstack{\\textbf{AIZynth} ($\\uparrow$)\\\\$W_1$ | $\\delta$}",
    ]

    lines = []
    lines.append("% Auto-generated by calculate_chemical_quality_distribution_tests.py")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Distributional shift and effect size of generated molecules relative to experimental reference series across the tractable benchmark targets (DPP4, ITK, PEPCK, CHK1, TTK). Values report macro-averaged Wasserstein distance ($W_1$, distribution divergence in natural metric units) and Cliff's Delta ($\\delta$, directional effect size $\\in [-1, 1]$) across 5 independent experimental runs (mean $\\pm$ standard deviation across seeds). Positive $\\delta$ indicates generated molecules score higher than the reference set; negative indicates lower. For QED and AIZynthFinder ($\\uparrow$), $\\delta > 0$ represents quality enhancement; for SA, MolSkill, and STOPLIGHT ($\\downarrow$), $\\delta < 0$ represents quality enhancement. Individual runs with fewer than 10 EYR molecules are excluded from seed-level averages to ensure statistical validity.}")
    lines.append("\\label{tab:quality_vs_reference}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Model}} & " + " & ".join(metric_headers) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        row_cells = [f"\\textbf{{{model}}}"]
        for m in metrics:
            sub = df_tractable[(df_tractable["model"] == model) & (df_tractable["metric"] == m)]
            if sub.empty or np.isnan(sub["macro_wasserstein_dist"].values[0]):
                row_cells.append("N/A")
                continue

            w1 = sub["macro_wasserstein_dist"].values[0]
            w1_std = sub["macro_wasserstein_dist_std"].values[0]
            delta = sub["macro_cliffs_delta"].values[0]
            delta_std = sub["macro_cliffs_delta_std"].values[0]

            if m == "molskill_score":
                w1_str = f"{w1:.1f}"
            else:
                w1_str = f"{w1:.2f}"

            delta_sign = "+" if delta > 0 else ""
            cell_str = f"{w1_str} $\\mid$ {delta_sign}{delta:.2f}"
            row_cells.append(cell_str)

        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines) + "\n"
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"Generated LaTeX table snippet at {latex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate chemical quality distribution tests.")
    parser.add_argument(
        "--exps-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock/exps_upperbound"),
        help="Path to experiment results directory.",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock/reference_set_scores"),
        help="Path to reference set scores directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock/analysis_exps_upperbound"),
        help="Directory to save statistical test outputs.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum EYR compounds required in a run to compute distribution tests (default: 10).",
    )
    args = parser.parse_args()

    analyze_experiments(args.exps_dir, args.ref_dir, args.output_dir, min_samples=args.min_samples)


if __name__ == "__main__":
    main()
