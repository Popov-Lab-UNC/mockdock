#!/usr/bin/env python3
"""Generate publication-quality figures summarizing chemical quality statistical comparisons.

Generates:
1. figure_quality_effect_sizes.png / .svg:
   - Panel A: Heatmap of directional effect sizes (Cliff's Delta) vs Reference Set.
     (Color-coded: blue/green = quality improvement, orange/red = degradation, grey = neutral)
   - Panel B: Wasserstein Distance (W1) distribution divergence across models.
   - Panel C: EYR molecule yield across benchmark targets (highlighting low vs high yield targets).

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


def plot_quality_summary(
    macro_csv: Path,
    target_summary_csv: Path,
    output_dirs: list[Path],
) -> None:
    setup_plotting()
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    macro_df = pl.read_csv(macro_csv)
    target_df = pl.read_csv(target_summary_csv)

    # Filter for tractable targets macro-average
    tractable_macro = macro_df.filter(pl.col("scope") == "Tractable_5_Targets").to_pandas()
    if tractable_macro.empty:
        tractable_macro = macro_df.to_pandas()

    models = [m for m in MODEL_PLOT_ORDER if m != REFERENCE_SET_LABEL and m in tractable_macro["model"].unique()]
    models += [m for m in tractable_macro["model"].unique() if m not in MODEL_PLOT_ORDER and m != REFERENCE_SET_LABEL]

    metrics = ["qed", "sa", "molskill_score", "stoplight_score", "aizynthfinder_state_score"]
    metric_labels = ["QED (↑)", "SA Score (↓)", "MolSkill (↓)", "STOPLIGHT (↓)", "AIZynth (↑)"]

    # ── 1. Construct Superiority / Cliff's Delta Matrix ────────────────────────
    # superior_effect: positive = better than reference, negative = worse than reference
    effect_matrix = np.zeros((len(models), len(metrics)))
    w1_matrix = np.zeros((len(models), len(metrics)))

    for i, model in enumerate(models):
        for j, m in enumerate(metrics):
            sub = tractable_macro[(tractable_macro["model"] == model) & (tractable_macro["metric"] == m)]
            if not sub.empty and not np.isnan(sub["macro_cliffs_delta"].values[0]):
                effect_matrix[i, j] = sub["macro_cliffs_delta"].values[0]
                w1_matrix[i, j] = sub["macro_wasserstein_dist"].values[0]
            else:
                effect_matrix[i, j] = np.nan
                w1_matrix[i, j] = np.nan

    # ── 2. Create Multi-Panel Figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)

    # Panel A: Effect Size Heatmap (Raw Cliff's Delta vs Reference Set)
    ax_a = fig.add_subplot(gs[0])

    # Colormap: diverging from Red (negative) -> Light Gray (neutral) -> Blue (positive)
    cmap = LinearSegmentedColormap.from_list(
        "quality_diverging",
        ["#d73027", "#f46d43", "#fdae61", "#f7f7f7", "#abd9e9", "#74add1", "#4575b4"],
        N=256,
    )

    im = ax_a.imshow(effect_matrix, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")

    # Ticks & Labels: horizontal orientation to prevent overlap with colorbar
    ax_a.set_xticks(range(len(metrics)))
    ax_a.set_xticklabels(metric_labels, fontsize=10.5, fontweight="bold", rotation=0, ha="center")
    ax_a.set_yticks(range(len(models)))
    ax_a.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Generative Model", fontsize=12, fontweight="bold")

    # Annotate values inside cells
    for i in range(len(models)):
        for j in range(len(metrics)):
            val = effect_matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                text_color = "#888888"
            else:
                sign = "+" if val > 0 else ""
                text = f"{sign}{val:.2f}"
                text_color = "white" if abs(val) > 0.38 else "#111111"
            ax_a.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    ax_a.set_title("A", loc="left", fontsize=14, fontweight="bold", pad=10)

    # Colorbar with generous padding to completely eliminate any overlap with x-axis labels
    cbar = fig.colorbar(im, ax=ax_a, orientation="horizontal", pad=0.25, fraction=0.046, shrink=0.75)
    cbar.set_label(r"Cliff's Delta ($\delta$)", fontsize=10.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Panel B: Average EYR Yield per Target (Accounting for Low-Yield Targets)
    ax_b = fig.add_subplot(gs[1])

    # Calculate average EYR molecules for High-Yield vs Pocket-Constrained targets per model
    target_summary_df = target_df.to_pandas()
    high_yields = []
    constrained_yields = []

    for model in models:
        sub_hi = target_summary_df[(target_summary_df["model"] == model) & (target_summary_df["target"].isin(["CHK1", "DPP4", "ITK", "PEPCK", "TTK"]))]
        sub_co = target_summary_df[(target_summary_df["model"] == model) & (target_summary_df["target"].isin(["PptT", "VEGFR2"]))]

        y_hi = sub_hi["mean_n_gen"].mean() if not sub_hi.empty else 0.0
        y_co = sub_co["mean_n_gen"].mean() if not sub_co.empty else 0.0

        high_yields.append(y_hi)
        constrained_yields.append(y_co)

    y_pos = np.arange(len(models))
    height = 0.38

    bars1 = ax_b.barh(y_pos - height/2, high_yields, height, label="High-EYR Targets (CHK1, DPP4, ITK, PEPCK, TTK)", color="#3182bd", alpha=0.9)
    bars2 = ax_b.barh(y_pos + height/2, constrained_yields, height, label="Low-EYR for PromptSMILES (PptT, VEGFR2)", color="#e6550d", alpha=0.9)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Generative Model", fontsize=12, fontweight="bold")
    ax_b.invert_yaxis()  # Top-down matching heatmap
    ax_b.set_xlabel("Mean Effective Yield Rate (EYR) Molecules / Run", fontsize=11, fontweight="bold", labelpad=8)
    ax_b.set_title("B", loc="left", fontsize=14, fontweight="bold", pad=10)
    ax_b.legend(loc="upper right", frameon=True, fontsize=9.5, framealpha=0.95)
    ax_b.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax_b.set_axisbelow(True)

    # Add vertical line at threshold N_min = 10
    ax_b.axvline(10, color="gray", linestyle=":", linewidth=1.2)
    ax_b.text(12, len(models) - 0.6, "$N_{min} = 10$", color="#555555", fontsize=9, fontstyle="italic")

    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.22)

    # Save to all target directories
    file_stems = [
        "figure_quality_effect_sizes",
        "figureS5_quality_statistical_summary",
        "figureS8_quality_statistical_summary",
    ]
    for out_dir in output_dirs:
        for stem in file_stems:
            png_path = out_dir / f"{stem}.png"
            svg_path = out_dir / f"{stem}.svg"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(svg_path, bbox_inches="tight")
            print(f"Saved figure: {png_path}")

    plt.close(fig)


def main() -> None:
    macro_csv = ANALYSIS_DIR / "chemical_quality_stats_macro_summary.csv"
    target_summary_csv = ANALYSIS_DIR / "chemical_quality_stats_target_summary.csv"

    if not macro_csv.exists():
        print(f"Error: {macro_csv} not found.")
        sys.exit(1)

    output_dirs = [
        ANALYSIS_DIR / "figures",
        PAPER_FIG_DIR,
    ]

    plot_quality_summary(macro_csv, target_summary_csv, output_dirs)


if __name__ == "__main__":
    main()
