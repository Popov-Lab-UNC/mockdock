#!/usr/bin/env python3
"""Generate publication-quality figure showing chemical quality statistical comparisons

Outputs:
1. figure_quality_by_task_cliffs_delta.png / .svg:
   5 heatmaps (one per metric: QED, SA, MolSkill, STOPLIGHT, AIZynthFinder).
   Rows: 9 Models.
   Columns: 7 Benchmark Targets (CHK1, DPP4, ITK, PEPCK, PptT, TTK, VEGFR2).
   Colors: Diverging colormap (Blue = superior to reference, Red = worse than reference).
   Annotations: Cliff's delta values and sample size indicators.

2. figure_quality_by_task_wasserstein.png / .svg:
   5 heatmaps showing Wasserstein Distance (W1) distribution divergence across models and targets.

Outputs saved to:
- analysis_exps_upperbound/figures/
- mockdock_paper/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.experiment_utils import MODEL_PLOT_ORDER, REFERENCE_SET_LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis_exps_upperbound"
PAPER_FIG_DIR = PROJECT_ROOT.parent / "mockdock_paper" / "figures"

TARGETS_ORDER = ["CHK1", "DPP4", "ITK", "PEPCK", "PptT", "TTK", "VEGFR2"]

METRIC_CONFIGS = [
    {"name": "qed", "label": "QED (↑)", "higher_is_better": True, "panel": "A"},
    {"name": "sa", "label": "SA Score (↓)", "higher_is_better": False, "panel": "B"},
    {"name": "molskill_score", "label": "MolSkill (↓)", "higher_is_better": False, "panel": "C"},
    {"name": "stoplight_score", "label": "STOPLIGHT (↓)", "panel": "D", "higher_is_better": False},
    {"name": "aizynthfinder_state_score", "label": "AIZynthFinder (↑)", "panel": "E", "higher_is_better": True},
]


def setup_plotting() -> None:
    """Configure styling for publication-quality figures."""
    sns.set_theme(style="white", font="sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def plot_by_task_heatmaps(
    target_summary_csv: Path,
    pooled_csv: Path,
    output_dirs: list[Path],
) -> None:
    setup_plotting()
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pl.read_csv(target_summary_csv)
    pooled_df = pl.read_csv(pooled_csv)

    models = [m for m in MODEL_PLOT_ORDER if m != REFERENCE_SET_LABEL and m in target_df["model"].unique()]
    models += [m for m in target_df["model"].unique() if m not in MODEL_PLOT_ORDER and m != REFERENCE_SET_LABEL]

    # Colormap: diverging from Red (worse than reference) -> Neutral Gray -> Blue (better than reference)
    cmap_div = LinearSegmentedColormap.from_list(
        "quality_diverging",
        ["#d73027", "#f46d43", "#fdae61", "#f7f7f7", "#abd9e9", "#74add1", "#3182bd"],
        N=256,
    )

    # Colormap for Wasserstein: Sequential (light = close to reference, dark purple = large divergence)
    cmap_seq = sns.color_palette("mako_r", as_cmap=True)

    # ── 1. FIGURE 1: Cliff's Delta Directional Effect Size by Task ────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    axes_flat = axes.flatten()

    for idx, mcfg in enumerate(METRIC_CONFIGS):
        ax = axes_flat[idx]
        metric = mcfg["name"]
        higher_is_better = mcfg["higher_is_better"]

        matrix = np.zeros((len(models), len(TARGETS_ORDER)))
        annot_matrix = []

        for i, model in enumerate(models):
            row_annots = []
            for j, target in enumerate(TARGETS_ORDER):
                sub = target_df.filter(
                    (pl.col("model") == model)
                    & (pl.col("target") == target)
                    & (pl.col("metric") == metric)
                )

                if sub.is_empty():
                    matrix[i, j] = np.nan
                    row_annots.append("N/A")
                    continue

                row_dict = sub.to_dicts()[0]
                n_valid = row_dict["n_valid_seeds"]
                mean_n = row_dict["mean_n_gen"]

                if n_valid >= 2 and not np.isnan(row_dict["cliffs_delta_mean"]):
                    eff = row_dict["cliffs_delta_mean"]
                    matrix[i, j] = eff
                    sign = "+" if eff > 0 else ""
                    row_annots.append(f"{sign}{eff:.2f}")
                elif n_valid == 1 and not np.isnan(row_dict["cliffs_delta_mean"]):
                    eff = row_dict["cliffs_delta_mean"]
                    matrix[i, j] = eff
                    sign = "+" if eff > 0 else ""
                    row_annots.append(f"{sign}{eff:.2f}*")
                else:
                    # Check pooled value as fallback for sparse runs (e.g. A2C VEGFR2)
                    psub = pooled_df.filter(
                        (pl.col("model") == model)
                        & (pl.col("target") == target)
                        & (pl.col("metric") == metric)
                    )
                    if not psub.is_empty() and not np.isnan(psub["cliffs_delta"][0]):
                        eff = psub["cliffs_delta"][0]
                        matrix[i, j] = eff
                        sign = "+" if eff > 0 else ""
                        row_annots.append(f"[{sign}{eff:.2f}]")
                    else:
                        matrix[i, j] = np.nan
                        row_annots.append("N/A")

            annot_matrix.append(row_annots)

        im = ax.imshow(matrix, cmap=cmap_div, vmin=-0.8, vmax=0.8, aspect="auto")

        ax.set_title(f"{mcfg['panel']}   {mcfg['label']}", loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xticks(range(len(TARGETS_ORDER)))
        ax.set_xticklabels(TARGETS_ORDER, fontsize=10, fontweight="bold")

        # Show model names on all subplots so each panel is self-contained
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9.5, fontweight="bold")
        if idx % 3 == 0:
            ax.set_ylabel("Generative Model", fontsize=11, fontweight="bold")

        if idx >= 3:
            ax.set_xlabel("Benchmark Target", fontsize=10.5, fontweight="bold")

        # Annotate text
        for i in range(len(models)):
            for j in range(len(TARGETS_ORDER)):
                text = annot_matrix[i][j]
                val = matrix[i, j]
                text_color = "white" if (not np.isnan(val) and abs(val) > 0.45) else "#111111"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color)

    # 6th panel: Clean colorbar without popup text box
    ax_legend = axes_flat[5]
    ax_legend.axis("off")

    # Inset colorbar inside 6th panel
    cbar_ax = ax_legend.inset_axes([0.15, 0.44, 0.70, 0.12])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Cliff's Delta ($\delta$)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)

    fig.tight_layout()

    file_stems = [
        "figure_quality_by_task_cliffs_delta",
        "figureS7_quality_by_task_cliffs_delta",
    ]
    for out_dir in output_dirs:
        for stem in file_stems:
            p_png = out_dir / f"{stem}.png"
            p_svg = out_dir / f"{stem}.svg"
            fig.savefig(p_png, bbox_inches="tight")
            fig.savefig(p_svg, bbox_inches="tight")
            print(f"Saved Cliff's Delta by task figure: {p_png}")

    plt.close(fig)

    # ── 2. FIGURE 2: Wasserstein Distance (W1) by Task ────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    axes2_flat = axes2.flatten()

    for idx, mcfg in enumerate(METRIC_CONFIGS):
        ax = axes2_flat[idx]
        metric = mcfg["name"]

        matrix_w1 = np.zeros((len(models), len(TARGETS_ORDER)))
        annot_matrix_w1 = []

        for i, model in enumerate(models):
            row_annots = []
            for j, target in enumerate(TARGETS_ORDER):
                sub = target_df.filter(
                    (pl.col("model") == model)
                    & (pl.col("target") == target)
                    & (pl.col("metric") == metric)
                )

                if sub.is_empty():
                    matrix_w1[i, j] = np.nan
                    row_annots.append("N/A")
                    continue

                row_dict = sub.to_dicts()[0]
                n_valid = row_dict["n_valid_seeds"]

                if n_valid >= 1 and not np.isnan(row_dict["wasserstein_dist_mean"]):
                    w1 = row_dict["wasserstein_dist_mean"]
                    matrix_w1[i, j] = w1
                    fmt = ".1f" if metric == "molskill_score" else ".2f"
                    row_annots.append(f"{w1:{fmt}}")
                else:
                    psub = pooled_df.filter(
                        (pl.col("model") == model)
                        & (pl.col("target") == target)
                        & (pl.col("metric") == metric)
                    )
                    if not psub.is_empty() and not np.isnan(psub["wasserstein_dist"][0]):
                        w1 = psub["wasserstein_dist"][0]
                        matrix_w1[i, j] = w1
                        fmt = ".1f" if metric == "molskill_score" else ".2f"
                        row_annots.append(f"[{w1:{fmt}}]")
                    else:
                        matrix_w1[i, j] = np.nan
                        row_annots.append("N/A")

            annot_matrix_w1.append(row_annots)

        im2 = ax.imshow(matrix_w1, cmap="magma_r", aspect="auto")

        ax.set_title(f"{mcfg['panel']}   {mcfg['label']}", loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xticks(range(len(TARGETS_ORDER)))
        ax.set_xticklabels(TARGETS_ORDER, fontsize=10, fontweight="bold")

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9.5, fontweight="bold")
        if idx % 3 == 0:
            ax.set_ylabel("Generative Model", fontsize=11, fontweight="bold")

        if idx >= 3:
            ax.set_xlabel("Benchmark Target", fontsize=10.5, fontweight="bold")

        # Annotate text
        v_max = np.nanmax(matrix_w1) if not np.all(np.isnan(matrix_w1)) else 1.0
        for i in range(len(models)):
            for j in range(len(TARGETS_ORDER)):
                text = annot_matrix_w1[i][j]
                val = matrix_w1[i, j]
                text_color = "white" if (not np.isnan(val) and val > 0.6 * v_max) else "#111111"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color)

    ax2_legend = axes2_flat[5]
    ax2_legend.axis("off")

    fig2.tight_layout()

    file_stems = [
        "figure_quality_by_task_wasserstein",
        "figureS6_quality_by_task_wasserstein",
    ]
    for out_dir in output_dirs:
        for stem in file_stems:
            p_png = out_dir / f"{stem}.png"
            p_svg = out_dir / f"{stem}.svg"
            fig2.savefig(p_png, bbox_inches="tight")
            fig2.savefig(p_svg, bbox_inches="tight")
            print(f"Saved Wasserstein by task figure: {p_png}")

    plt.close(fig2)


def main() -> None:
    macro_csv = ANALYSIS_DIR / "chemical_quality_stats_macro_summary.csv"
    target_summary_csv = ANALYSIS_DIR / "chemical_quality_stats_target_summary.csv"
    pooled_csv = ANALYSIS_DIR / "chemical_quality_stats_pooled.csv"

    if not target_summary_csv.exists() or not pooled_csv.exists():
        print(f"Error: summary CSVs not found in {ANALYSIS_DIR}")
        sys.exit(1)

    output_dirs = [
        ANALYSIS_DIR / "figures",
        PAPER_FIG_DIR,
    ]

    plot_by_task_heatmaps(target_summary_csv, pooled_csv, output_dirs)


if __name__ == "__main__":
    main()
