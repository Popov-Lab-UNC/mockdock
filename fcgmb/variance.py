import os
import subprocess
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy.stats import pearsonr, spearmanr

def get_pactivity(df: pl.DataFrame, config_path: Path):
    """
    Robustly convert activity to pActivity using ChEMBL pValue or YAML units.
    """
    if "pchembl_value" in df.columns:
        return df.get_column("pchembl_value").to_numpy(), "pActivity (ChEMBL pValue)"
    
    # Fallback to standard_value + units from YAML
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        units = config.get("activity_units", "nM")
    except:
        units = "nM"
        
    activities = df.get_column("standard_value").to_numpy()
    unit_offsets = {"nM": 9, "uM": 6, "mM": 3, "M": 0}
    offset = unit_offsets.get(units, 9)
    p_activities = offset - np.log10(activities + 1e-12)
    return p_activities, f"pActivity (-log10 {units})"

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

    for config_path in configs:
        print(f"\n>>> Processing system: {config_path.stem}")
        
        # 1. Run initialization (retrieve + grid) ONCE
        init_run_dir = run_base_dir / "init"
        subprocess.run([
            "python", workflow_script,
            "--config", str(config_path),
            "--stage", "retrieve",
            "--run-dir", str(init_run_dir)
        ])
        
        subprocess.run([
            "python", workflow_script,
            "--config", str(config_path),
            "--stage", "grid",
            "--run-dir", str(init_run_dir)
        ])

        # 2. Run docking n_iterations times
        for i in range(1, n_iterations + 1):
            iter_run_dir = run_base_dir / f"run_{i}"
            print(f"  -> Iteration {i}/{n_iterations}...")
            
            config = yaml.safe_load(config_path.read_text())
            target_id = config.get("target_id")
            pdb_id = config.get("pdb_id")
            target_pdb_name = f"{target_id}_{pdb_id}"
            
            # Re-use grid and data from init
            src_grid = init_run_dir / target_pdb_name / "grid"
            dst_target_dir = iter_run_dir / target_pdb_name
            dst_target_dir.mkdir(parents=True, exist_ok=True)
            dst_grid = dst_target_dir / "grid"
            
            if src_grid.exists() and not dst_grid.exists():
                try:
                    os.symlink(src_grid.resolve(), dst_grid)
                except OSError:
                    import shutil
                    shutil.copytree(src_grid, dst_grid)
            
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

            # Run docking
            subprocess.run([
                "python", workflow_script,
                "--config", str(config_path),
                "--stage", "docking",
                "--run-dir", str(iter_run_dir)
            ])

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
            subset = df.select(["canonical_smiles", "docking_score", "standard_value", 
                                "pchembl_value" if "pchembl_value" in df.columns else "standard_value"])
            subset = subset.rename({"docking_score": f"score_{i}"})
            if merged is None:
                merged = subset
            else:
                cols_to_drop = [c for c in ["standard_value", "pchembl_value"] if c in merged.columns]
                merged = merged.join(subset.drop(cols_to_drop), on="canonical_smiles", how="inner")

        score_cols = [c for c in merged.columns if c.startswith("score_")]
        valid_mask = merged.select([(pl.col(c).is_not_null()) & (pl.col(c) < 900) for c in score_cols]).reduce(lambda a, b: a & b)
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
