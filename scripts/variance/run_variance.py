#!/usr/bin/env python3
"""
Standalone script to run variance tests for the docking workflow.
Relocated from legacy variance module to allow cleaner package structure.
"""

# Standard library imports
import os
import subprocess
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PALETTE = {
    "periwinkle": "#B8B8FF",
    "light_green": "#90EE90",
    "light_blue": "#0072B2",
    "orange": "#FF7F00",
    "soft_pink": "#E89EB8",
    "caramel": "#C08552",
}


def set_publication_style():
    """Set publication-ready defaults for variance plots."""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def get_pactivity(df: pl.DataFrame, config_path: Path):
    """
    Use ChEMBL pValue only; error if missing.
    """
    if "pchembl_value" not in df.columns:
        raise RuntimeError("Missing pchembl_value; cannot compute pActivity.")
    return df.get_column("pchembl_value").to_numpy(), "pActivity (ChEMBL pValue)"


def _load_workflow_config(config_path: Path) -> dict:
    if config_path.suffix.lower() == ".toml":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        benchmark_name = config.get("benchmark_name", config_path.stem)
        ligand_csv_path = config.get("ligand_csv_path")
        if ligand_csv_path:
            ligand_path = Path(ligand_csv_path)
            if not ligand_path.is_absolute():
                ligand_path = (config_path.parent / ligand_path).resolve()
            config["ligand_csv_path"] = str(ligand_path)
        else:
            candidate = (
                Path(__file__).resolve().parents[2] / "src" / "mockdock" / "bioactivity_data" / f"{benchmark_name}.csv"
            )
            if not candidate.exists():
                raise FileNotFoundError(
                    f"No ligand_csv_path in {config_path} and packaged CSV missing at {candidate}"
                )
            config["ligand_csv_path"] = str(candidate.resolve())
        return config
    return yaml.safe_load(config_path.read_text())


def _resolve_config_paths(
    config_dir: Path,
    single_config: Path | None,
) -> list[Path]:
    if single_config:
        return [single_config]
    configs = sorted(config_dir.glob("*.yaml"))
    configs.extend(sorted(config_dir.glob("*.toml")))
    return configs


def run_variance_tests(
    config_dir: Path = Path("configs"),
    single_config: Path | None = None,
    run_base_dir: Path = Path("variance_runs"),
    n_iterations: int = 5,
    workflow_script: str = str(Path(__file__).resolve().parents[1] / "docking" / "run_workflow.py"),
):
    """
    Run docking multiple times for each benchmark to test variance.
    """
    run_base_dir.mkdir(exist_ok=True, parents=True)
    configs = _resolve_config_paths(config_dir, single_config)

    if not configs:
        print(f"No configs found in {config_dir}")
        return

    print("\n" + "=" * 60)
    print(f"VARIANCE BENCHMARK: Running {len(configs)} systems, {n_iterations} iterations each")
    print(f"Base Directory: {run_base_dir}")
    print("=" * 60 + "\n")

    for config_path in configs:
        config_data = _load_workflow_config(config_path)
        pdb_id = config_data.get("pdb_id", config_path.stem)
        print(f"[{pdb_id}] Initializing maps and data...")

        # 1. Run initialization (retrieve + grid) ONCE
        init_run_dir = run_base_dir / "init"
        try:
            subprocess.run(
                [
                    sys.executable,
                    workflow_script,
                    "--config",
                    str(config_path.resolve()),
                    "--stage",
                    "retrieve",
                    "--run-dir",
                    str(init_run_dir),
                    "--quiet",
                ],
                check=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    workflow_script,
                    "--config",
                    str(config_path.resolve()),
                    "--stage",
                    "grid",
                    "--run-dir",
                    str(init_run_dir),
                    "--quiet",
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"[{pdb_id}] Initialization FAILED. Skipping this system.")
            continue

        print(f"[{pdb_id}] Running {n_iterations} docking iterations:")
        # 2. Run docking n_iterations times
        for i in range(1, n_iterations + 1):
            iter_run_dir = run_base_dir / f"run_{i}"
            print(f"  -> Iteration {i}/{n_iterations}...", end=" ", flush=True)

            config = _load_workflow_config(config_path)
            target_id = config.get("target_id")
            pdb_id = config.get("pdb_id")
            target_pdb_name = f"{target_id}_{pdb_id}"

            # Re-use grid and data from init
            src_grid_dir = init_run_dir / target_pdb_name
            dst_target_dir = iter_run_dir / target_pdb_name
            dst_target_dir.mkdir(parents=True, exist_ok=True)

            if src_grid_dir.exists():
                for grid_file in src_grid_dir.iterdir():
                    if not grid_file.is_file():
                        continue
                    dst_file = dst_target_dir / grid_file.name
                    if dst_file.exists():
                        continue
                    try:
                        os.symlink(grid_file.resolve(), dst_file)
                    except OSError:
                        import shutil

                        shutil.copy2(grid_file, dst_file)

            doc_id = config.get("doc_id")
            src_work = init_run_dir / target_pdb_name / str(doc_id)
            dst_work = dst_target_dir / str(doc_id)
            dst_work.mkdir(parents=True, exist_ok=True)

            # Keep cleaned-data filename consistent with workflow naming.
            prefix = f"{target_id}_{pdb_id}_{doc_id}"
            assay_id = config.get("assay_id")
            if assay_id:
                prefix += f"_{assay_id}"
            data_file = f"{prefix}_cleaned_data.csv"

            if (src_work / data_file).exists() and not (dst_work / data_file).exists():
                try:
                    os.symlink((src_work / data_file).resolve(), dst_work / data_file)
                except OSError:
                    import shutil

                    shutil.copy2(src_work / data_file, dst_work / data_file)

            try:
                subprocess.run(
                    [
                        sys.executable,
                        workflow_script,
                        "--config",
                        str(config_path.resolve()),
                        "--stage",
                        "docking",
                        "--run-dir",
                        str(iter_run_dir),
                        "--quiet",
                    ],
                    check=True,
                )
                print("DONE.")
            except subprocess.CalledProcessError:
                print("FAILED.")
            except Exception as e:
                print(f"ERROR: {e}")


def analyze_variance_results(
    run_base_dir: Path = Path("variance_runs"),
    config_dir: Path = Path("configs"),
    output_dir: Path = Path("variance_analysis"),
):
    """
    Analyze results from variance tests and generate plots.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    runs = sorted(list(run_base_dir.glob("run_*")))
    if not runs:
        print("No runs found. Skipping analysis.")
        return

    system_data = {}
    print(f"Scanning {len(runs)} run directories...")
    for run_dir in runs:
        for csv_path in run_dir.glob("**/*_results_full.csv"):
            doc_id = csv_path.parent.name
            target_pdb = csv_path.parent.parent.name
            system_key = f"{target_pdb}_{doc_id}"

            if system_key not in system_data:
                system_data[system_key] = []
            system_data[system_key].append(pl.read_csv(csv_path))

    for system_key, df_list in system_data.items():
        if len(df_list) < 2:
            continue

        print(f"\nAnalyzing {system_key} ({len(df_list)} runs)...")

        # Merge docking scores
        merged = pl.DataFrame()
        for i, df in enumerate(df_list):
            if "pchembl_value" not in df.columns:
                raise RuntimeError("Missing pchembl_value in results; cannot analyze variance.")
            subset = df.select(["canonical_smiles", "docking_score", "pchembl_value"])
            subset = subset.rename({"docking_score": f"score_{i}"})
            if merged.is_empty():
                merged = subset
            else:
                cols_to_drop = [c for c in ["pchembl_value"] if c in merged.columns]
                merged = merged.join(subset.drop(cols_to_drop), on="canonical_smiles", how="inner")

        score_cols = [c for c in merged.columns if c.startswith("score_")]
        valid_mask = pl.all_horizontal(
            [(pl.col(c).is_not_null()) & (pl.col(c) < 900) for c in score_cols]
        )
        clean_merged = merged.filter(valid_mask)

        if clean_merged.is_empty():
            print(f"  No valid scores found for {system_key}")
            continue

        # Calculate variance stats
        scores_matrix = clean_merged.select(score_cols).to_numpy()
        means = np.mean(scores_matrix, axis=1)
        stds = np.std(scores_matrix, axis=1)

        # Get pActivity (Y-axis)
        p_activities, activity_label = get_pactivity(clean_merged, config_dir)

        # Plot with true lower-quartile highlighting.
        q25_threshold = np.percentile(p_activities, 25)
        is_lower_quartile = p_activities <= q25_threshold

        set_publication_style()
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            means[~is_lower_quartile],
            p_activities[~is_lower_quartile],
            xerr=stds[~is_lower_quartile],
            fmt="o",
            color=PALETTE["light_blue"],
            ecolor=PALETTE["soft_pink"],
            alpha=0.7,
            capsize=3,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Analogs",
        )
        plt.errorbar(
            means[is_lower_quartile],
            p_activities[is_lower_quartile],
            xerr=stds[is_lower_quartile],
            fmt="o",
            color=PALETTE["orange"],
            ecolor=PALETTE["soft_pink"],
            alpha=0.8,
            capsize=3,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Model visible (Lower 25%)",
        )

        plt.title(f"Score Variance vs {activity_label}\nSystem: {system_key}")
        plt.xlabel("Mean Docking Score")
        plt.ylabel(activity_label)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
        svg_path = output_dir / f"{system_key}_variance_plot.svg"
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()
        print(f"  Analysis saved to {svg_path}")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Variance tests for Docking Workflow")
    parser.add_argument(
        "--config-dir", type=str, default="configs", help="Directory with YAML/TOML configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Single workflow config path (.yaml) or benchmark config path (.toml).",
    )
    parser.add_argument(
        "--run-dir", type=str, default="variance_runs", help="Base directory for runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="variance_analysis",
        help="Directory for analysis output",
    )
    parser.add_argument(
        "--n-iters", "--iterations", type=int, default=5, help="Number of iterations per system"
    )
    parser.add_argument("--workflow", type=str, help="Path to run_workflow.py script")
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only run analysis on existing results"
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    single_config = Path(args.config).resolve() if args.config else None
    run_base_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    if not args.analyze_only:
        workflow_script = (
            args.workflow if args.workflow else str(Path(__file__).resolve().parents[1] / "docking" / "run_workflow.py")
        )
        run_variance_tests(
            config_dir=config_dir,
            single_config=single_config,
            run_base_dir=run_base_dir,
            n_iterations=args.n_iters,
            workflow_script=workflow_script,
        )

    analyze_variance_results(
        run_base_dir=run_base_dir, config_dir=config_dir, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
