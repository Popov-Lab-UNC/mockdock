# Standard library imports
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr

def get_pactivity(df: pl.DataFrame, config_path: Path):
    """
    Use ChEMBL pValue only; error if missing.
    """
    if "pchembl_value" not in df.columns:
        raise RuntimeError("Missing pchembl_value; cannot compute pActivity.")
    return df.get_column("pchembl_value").to_numpy(), "pActivity (ChEMBL pValue)"

def run_variance_tests(
    config_dir: Path = Path("configs"),
    run_base_dir: Path = Path("variance_runs"),
    n_iterations: int = 5,
    workflow_script: str = "run_workflow.py"
):
    """
    Run docking multiple times for each benchmark to test variance.
    """
    run_base_dir.mkdir(exist_ok=True, parents=True)
    configs = list(config_dir.glob("*.yaml"))
    
    if not configs:
        print(f"No configs found in {config_dir}")
        return

    print(f"\n" + "="*60)
    print(f"VARIANCE BENCHMARK: Running {len(configs)} systems, {n_iterations} iterations each")
    print(f"Base Directory: {run_base_dir}")
    print("="*60 + "\n")

    for config_path in configs:
        config_data = yaml.safe_load(config_path.read_text())
        pdb_id = config_data.get("pdb_id", config_path.stem)
        print(f"[{pdb_id}] Initializing maps and data...")
        
        # 1. Run initialization (retrieve + grid) ONCE
        init_run_dir = run_base_dir / "init"
        try:
            subprocess.run([
                "python", workflow_script,
                "--config", str(config_path),
                "--stage", "retrieve",
                "--run-dir", str(init_run_dir),
                "--quiet"
            ], check=True)
            
            subprocess.run([
                "python", workflow_script,
                "--config", str(config_path),
                "--stage", "grid",
                "--run-dir", str(init_run_dir),
                "--quiet"
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"[{pdb_id}] Initialization FAILED. Skipping this system.")
            continue

        print(f"[{pdb_id}] Running {n_iterations} docking iterations:")
        # 2. Run docking n_iterations times
        for i in range(1, n_iterations + 1):
            iter_run_dir = run_base_dir / f"run_{i}"
            print(f"  -> Iteration {i}/{n_iterations}...", end=" ", flush=True)
            
            config = yaml.safe_load(config_path.read_text())
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
            
            prefix = f"{target_id}_{pdb_id}_{doc_id}"
            data_file = f"{prefix}_cleaned_data.csv"
            
            if (src_work / data_file).exists() and not (dst_work / data_file).exists():
                try:
                    os.symlink((src_work / data_file).resolve(), dst_work / data_file)
                except OSError:
                    import shutil
                    shutil.copy2(src_work / data_file, dst_work / data_file)

            try:
                subprocess.run([
                    "python", workflow_script,
                    "--config", str(config_path),
                    "--stage", "docking",
                    "--run-dir", str(iter_run_dir),
                    "--quiet"
                ], check=True)
                print("DONE.")
            except subprocess.CalledProcessError:
                print("FAILED.")
            except Exception as e:
                print(f"ERROR: {e}")

def analyze_variance_results(
    run_base_dir: Path = Path("variance_runs"),
    config_dir: Path = Path("configs"),
    output_dir: Path = Path("variance_analysis")
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
        merged = None
        for i, df in enumerate(df_list):
            if "pchembl_value" not in df.columns:
                raise RuntimeError("Missing pchembl_value in results; cannot analyze variance.")
            subset = df.select(["canonical_smiles", "docking_score", "pchembl_value"])
            subset = subset.rename({"docking_score": f"score_{i}"})
            if merged is None:
                merged = subset
            else:
                cols_to_drop = [c for c in ["pchembl_value"] if c in merged.columns]
                merged = merged.join(subset.drop(cols_to_drop), on="canonical_smiles", how="inner")

        score_cols = [c for c in merged.columns if c.startswith("score_")]
        valid_mask = pl.all_horizontal([(pl.col(c).is_not_null()) & (pl.col(c) < 900) for c in score_cols])
        clean_merged = merged.filter(valid_mask)
        
        if clean_merged.is_empty():
            print(f"  No valid scores found for {system_key}")
            continue

        # Calculate variance stats
        scores_matrix = clean_merged.select(score_cols).to_numpy()
        means = np.mean(scores_matrix, axis=1)
        stds = np.std(scores_matrix, axis=1)
        
        # Get pActivity (Y-axis)
        config_path = config_dir / f"{system_key}.yaml"
        p_activities, activity_label = get_pactivity(clean_merged, config_path)

        # Plot
        plt.figure(figsize=(10, 7))
        sns.set_style("whitegrid")
        plt.errorbar(means, p_activities, xerr=stds, fmt='o', color='#2c7bb6', ecolor='#d7191c', 
                    alpha=0.6, capsize=3, markersize=5, label='Ligand Variance (Multiple Runs)')
        
        if len(means) > 1:
            p_corr, _ = pearsonr(means, p_activities)
            s_corr, _ = spearmanr(means, p_activities)
            plt.text(0.05, 0.95, f"N: {len(means)}\nPearson: {p_corr:.3f}\nSpearman: {s_corr:.3f}", 
                     transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f"Score Variance vs {activity_label}\nSystem: {system_key}", fontsize=13)
        plt.xlabel("Mean Docking Score (kcal/mol)", fontsize=11)
        plt.ylabel(activity_label, fontsize=11)
        plt.savefig(output_dir / f"{system_key}_variance_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Analysis saved to {output_dir / f'{system_key}_variance_plot.png'}")
