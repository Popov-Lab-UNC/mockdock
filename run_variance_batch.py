
import os
import subprocess
import polars as pl
from pathlib import Path
import yaml
import time

def run_variance_batch():
    config_dir = Path("/work/users/s/h/shuhang/benchmark/best_pl_system_configs")
    run_base_dir = Path("/work/users/s/h/shuhang/benchmark/variance_runs")
    run_base_dir.mkdir(exist_ok=True)

    configs = list(config_dir.glob("*.yaml"))
    
    # We'll run 5 iterations
    n_iterations = 5

    for config_path in configs:
        print(f"\n>>> Processing system: {config_path.stem}")
        
        # 1. Run initialization (retrieve + grid) ONCE
        # We use a shared run-dir for the initialization
        init_run_dir = run_base_dir / "init"
        subprocess.run([
            "python", "run_workflow.py",
            "--config", str(config_path),
            "--stage", "retrieve",
            "--run-dir", str(init_run_dir)
        ], cwd="/work/users/s/h/shuhang/benchmark")
        
        subprocess.run([
            "python", "run_workflow.py",
            "--config", str(config_path),
            "--stage", "grid",
            "--run-dir", str(init_run_dir)
        ], cwd="/work/users/s/h/shuhang/benchmark")

        # 2. Run docking 5 times
        for i in range(1, n_iterations + 1):
            iter_run_dir = run_base_dir / f"run_{i}"
            print(f"  -> Iteration {i}/{n_iterations}...")
            
            # Since we want to reuse the grid from init_run_dir,
            # we need to make sure run_workflow.py can find it.
            # Usually, run_workflow.py looks in <run-dir>/<target_id>_<pdb_id>/grid
            # If we point --run-dir to run_{i}, it might not find the grid unless we copy it or symlink it.
            # Alternately, we can just run docking and specify where the output goes.
            
            # Let's check how run_workflow.py handles grid discovery.
            # If args.run_dir is set, it sets target_pdb_dir = run_base / target_pdb_name
            # and grid_base_dir = target_pdb_dir.
            
            # So if we want to reuse grid from 'init', we should symlink it.
            
            config = yaml.safe_load(config_path.read_text())
            target_id = config.get("target_id")
            pdb_id = config.get("pdb_id")
            target_pdb_name = f"{target_id}_{pdb_id}"
            
            src_grid = init_run_dir / target_pdb_name / "grid"
            dst_target_dir = iter_run_dir / target_pdb_name
            dst_target_dir.mkdir(parents=True, exist_ok=True)
            dst_grid = dst_target_dir / "grid"
            
            if src_grid.exists() and not dst_grid.exists():
                os.symlink(src_grid.resolve(), dst_grid)
            
            # We also need the cleaned_data.csv in the work_dir
            doc_id = config.get("doc_id")
            src_work = init_run_dir / target_pdb_name / str(doc_id)
            dst_work = dst_target_dir / str(doc_id)
            dst_work.mkdir(parents=True, exist_ok=True)
            
            # Files to link: <prefix>_cleaned_data.csv
            # Prefix logic in run_workflow.py: f"{target_id}_{pdb_id}_{doc_id}"
            prefix = f"{target_id}_{pdb_id}_{doc_id}"
            data_file = f"{prefix}_cleaned_data.csv"
            
            if (src_work / data_file).exists() and not (dst_work / data_file).exists():
                os.symlink((src_work / data_file).resolve(), dst_work / data_file)

            # Run docking
            subprocess.run([
                "python", "run_workflow.py",
                "--config", str(config_path),
                "--stage", "docking",
                "--run-dir", str(iter_run_dir)
            ], cwd="/work/users/s/h/shuhang/benchmark")

if __name__ == "__main__":
    run_variance_batch()
