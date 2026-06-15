#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _get_pactivity(df: pl.DataFrame, config_path: Path | None = None):
    """Use ChEMBL pValue only; error if missing."""
    if "pchembl_value" not in df.columns:
        raise RuntimeError("Missing pchembl_value; cannot compute pActivity.")
    return df.get_column("pchembl_value").to_numpy(), "pActivity (ChEMBL pValue)"


PALETTE = {
    "periwinkle": "#B8B8FF",
    "light_green": "#90EE90",
    "light_blue": "#0072B2",
    "orange": "#FF7F00",
    "soft_pink": "#E89EB8",
    "caramel": "#C08552",
}


def set_publication_style():
    """Sets a consistent style for publication-ready figures."""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _load_crystal_mapping(mapping_path: Path) -> dict[str, dict[str, str]]:
    """Loads system_key -> {molecule_id, label} mapping."""
    if not mapping_path.exists():
        return {}
    df = pl.read_csv(mapping_path)
    mapping = {}
    for row in df.to_dicts():
        label = row.get("label")
        if label is None or str(label).strip() == "":
            label = "Crystal ligand"
        mapping[row["system_key"]] = {
            "molecule_id": str(row["molecule_chembl_id"]),
            "label": str(label),
        }
    return mapping


def _load_config(config_path: Path) -> dict:
    if config_path.suffix.lower() == ".toml":
        with config_path.open("rb") as f:
            return tomllib.load(f)
    return yaml.safe_load(config_path.read_text())


def _system_keys_from_config(cfg: dict) -> list[str]:
    target_id = cfg.get("target_id")
    pdb_id = cfg.get("pdb_id")
    doc_id = cfg.get("doc_id")
    assay_id = cfg.get("assay_id")
    if not all([target_id, pdb_id, doc_id]):
        return []

    keys = [f"{target_id}_{pdb_id}_{doc_id}"]
    if assay_id:
        keys.append(f"{target_id}_{pdb_id}_{doc_id}_{assay_id}")
    return keys


def _build_system_key_to_config(config_dir: Path) -> dict[str, Path]:
    """Map system_key (target_id_pdb_id_doc_id_assay_id) to config file path."""
    mapping = {}
    config_paths = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.toml"))
    for config_path in config_paths:
        try:
            cfg = _load_config(config_path)
            for key in _system_keys_from_config(cfg):
                mapping[key] = config_path
        except Exception:
            continue
    return mapping


def _iter_run_dirs(runs_dir: Path) -> list[Path]:
    return sorted([p for p in runs_dir.glob("run_*") if p.is_dir()])


def _iter_results(run_dir: Path) -> list[tuple[str, Path]]:
    # Layout: run_dir/<target_id>_<pdb_id>/<doc_id>/<target_id>_<pdb_id>_<doc_id>_<assay_id>_results.csv
    # system_key is derived from the CSV filename stem (strip trailing "_results")
    results = []
    for csv_path in run_dir.glob("**/*_results.csv"):
        try:
            system_key = csv_path.stem[: -len("_results")]
            results.append((system_key, csv_path))
        except Exception:
            continue
    return results


def _valid_score_mask(df: pl.DataFrame, score_cols: list[str]) -> pl.Expr:
    exprs = [
        (pl.col(c).is_not_null()) & (pl.col(c).is_finite()) & (pl.col(c) < 900) for c in score_cols
    ]
    return pl.fold(pl.lit(True), lambda acc, x: acc & x, exprs)


def _short_system_label(system_key: str) -> str:
    """Use only first CHEMBL ID and PDB ID from a full system key."""
    parts = system_key.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return system_key


SUMMARY_SCHEMA = {
    "system": pl.Utf8,
    "n_runs": pl.Int64,
    "pearson_mean": pl.Float64,
    "pearson_std": pl.Float64,
    "r2_mean": pl.Float64,
    "r2_std": pl.Float64,
    "spearman_mean": pl.Float64,
    "spearman_std": pl.Float64,
}


def _empty_summary_df() -> pl.DataFrame:
    return pl.DataFrame(schema=SUMMARY_SCHEMA)


def _merge_existing_summary(stats_df: pl.DataFrame, stats_csv: Path) -> pl.DataFrame:
    """Upsert newly computed systems into an existing summary CSV."""
    if not stats_csv.exists():
        return stats_df.sort("spearman_mean", descending=True)

    existing = pl.read_csv(stats_csv)
    missing_cols = [col for col in SUMMARY_SCHEMA if col not in existing.columns]
    if missing_cols:
        raise RuntimeError(f"Existing summary is missing columns: {missing_cols}")

    existing = existing.select(list(SUMMARY_SCHEMA))
    if not stats_df.is_empty():
        new_systems = stats_df.get_column("system").to_list()
        existing = existing.filter(~pl.col("system").is_in(new_systems))

    return pl.concat([existing, stats_df.select(list(SUMMARY_SCHEMA))]).sort(
        "spearman_mean", descending=True
    )


def _compute_system_correlations(df: pl.DataFrame, config_path: Path | None) -> tuple[float, float]:
    if df.is_empty() or "docking_score" not in df.columns:
        return float("nan"), float("nan")
    p_activities, _ = _get_pactivity(df, config_path)
    scores = df.get_column("docking_score").to_numpy()
    mask = np.isfinite(scores) & np.isfinite(p_activities) & (scores < 900)
    if np.sum(mask) < 2:
        return float("nan"), float("nan")
    pearson = pearsonr(scores[mask], p_activities[mask])[0]
    spearman = spearmanr(scores[mask], p_activities[mask]).correlation
    return pearson, spearman


def plot_run_barplot(runs_dir: Path, config_dir: Path, output_dir: Path) -> Path:
    run_dirs = _iter_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_dir}")

    system_key_to_config = _build_system_key_to_config(config_dir)
    system_corrs: dict[str, list[float]] = {}
    for run_dir in run_dirs:
        for system_key, csv_path in _iter_results(run_dir):
            config_path = system_key_to_config.get(system_key)
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

    stats_csv = output_dir / "run_correlation_summary.csv"
    stats_df = _empty_summary_df() if not system_stats else pl.DataFrame(system_stats)
    stats_df = _merge_existing_summary(stats_df, stats_csv)
    stats_df.write_csv(stats_csv)

    out_path = output_dir / "system_mean_std_barplot.svg"
    set_publication_style()
    plt.figure(figsize=(12, 6))
    summary_stats = stats_df.to_dicts()
    if summary_stats:
        sorted_stats = sorted(summary_stats, key=lambda r: r["spearman_mean"], reverse=True)
        x = np.arange(len(sorted_stats))
        means = [r["spearman_mean"] for r in sorted_stats]
        stds = [r["spearman_std"] for r in sorted_stats]
        labels = [_short_system_label(r["system"]) for r in sorted_stats]

        plt.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=PALETTE["light_blue"],
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        )
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Mean Spearman Correlation")
        plt.title("System Performance Stability Across Runs")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        plt.text(
            0.5,
            0.5,
            "No valid systems for correlation barplot",
            ha="center",
            va="center",
            fontsize=14,
            color=PALETTE["caramel"],
        )
        plt.xticks([])
        plt.yticks([])
        plt.title("System Performance Stability Across Runs")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_system_variance(
    runs_dir: Path,
    config_dir: Path,
    output_dir: Path,
    mapping: dict[str, dict[str, str]] = None,
) -> list[Path]:
    mapping = mapping or {}
    system_key_to_config = _build_system_key_to_config(config_dir)
    run_dirs = _iter_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_dir}")

    # Collect per-system data across runs
    system_data: dict[str, list[pl.DataFrame]] = {}
    for run_dir in run_dirs:
        for system_key, csv_path in _iter_results(run_dir):
            df = pl.read_csv(csv_path)
            system_data.setdefault(system_key, []).append(df)

    plot_paths = []
    for system_key, df_list in system_data.items():
        if len(df_list) < 2:
            continue

        id_col = None
        for df in df_list:
            if "molecule_chembl_id" in df.columns:
                id_col = "molecule_chembl_id"
                break
            if "canonical_smiles" in df.columns:
                id_col = "canonical_smiles"
                break

        if id_col is None or any("pchembl_value" not in df.columns for df in df_list):
            continue

        merged = None
        for i, df in enumerate(df_list):
            columns = [id_col, "docking_score", "pchembl_value"]
            subset = df.select(columns).rename({"docking_score": f"score_{i}"})
            if merged is None:
                merged = subset
            else:
                merged = merged.join(subset.drop("pchembl_value"), on=id_col, how="inner")

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

        config_path = system_key_to_config.get(system_key)
        p_activities, activity_label = _get_pactivity(clean, config_path)

        # Implementation for true quantile shading and crystal labeling
        set_publication_style()
        plt.figure(figsize=(12, 8))

        # Calculate true 25th percentile quantile
        q25_threshold = np.percentile(p_activities, 25)
        is_lower_quartile = p_activities <= q25_threshold

        # Draw error bars for both groups with legend support
        plt.errorbar(
            means[~is_lower_quartile],
            p_activities[~is_lower_quartile],
            xerr=stds[~is_lower_quartile],
            fmt="o",
            color=PALETTE["light_blue"],
            alpha=0.6,
            capsize=3,
            markersize=9,
            label="Analogs",
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        plt.errorbar(
            means[is_lower_quartile],
            p_activities[is_lower_quartile],
            xerr=stds[is_lower_quartile],
            fmt="o",
            color=PALETTE["orange"],
            alpha=0.7,
            capsize=3,
            markersize=9,
            label="Model visible (Lower pActivity 25%)",
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        # Highlight crystal ligand if in mapping
        crystal_info = mapping.get(system_key)
        crystal_label_missing = True
        if crystal_info:
            target_id = str(crystal_info["molecule_id"])
            mask = clean.get_column(id_col) == target_id
            if mask.any():
                idx = np.where(mask.to_numpy())[0][0]
                crystal_label_missing = False
                plt.scatter(
                    means[idx],
                    p_activities[idx],
                    color=PALETTE["caramel"],
                    s=95,
                    edgecolor="black",
                    zorder=10,
                    label=crystal_info["label"],
                    marker="D",
                )
                # Crystal ligand appears in legend only (no on-plot text).

        plt.title(f"Docking Score Stability vs {activity_label}\nSystem: {system_key}")
        plt.xlabel("Mean Docking Score")
        plt.ylabel(activity_label)
        if crystal_label_missing:
            plt.text(
                0.99,
                0.01,
                "Crystal ligand mapping: missing/unmatched",
                transform=plt.gca().transAxes,
                ha="right",
                va="bottom",
                color=PALETTE["caramel"],
                fontsize=10,
            )
        plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

        out_path = output_dir / f"{system_key}_variance_plot.svg"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        plot_paths.append(out_path)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(
        description="Analyze variance runs across multiple run_* directories"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="variance_runs",
        help="Base variance runs directory",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="src/mockdock/configs",
        help="Directory with system YAML/TOML configs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="variance_runs",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="variance_runs/crystal_ligand_mapping.csv",
        help="CSV mapping crystal ligands",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping)

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_crystal_mapping(mapping_path)
    if not mapping:
        print(f"Warning: No crystal ligand mapping found at {mapping_path}")

    barplot_path = plot_run_barplot(runs_dir, config_dir, output_dir)
    plot_system_variance(runs_dir, config_dir, output_dir, mapping=mapping)

    print(f"Saved bar plot to: {barplot_path}")
    print(f"Saved per-system variance plots to: {output_dir}")


if __name__ == "__main__":
    main()
