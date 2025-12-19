from docking_benchmark import (
    fetch_chembl_data, 
    AutoDockGPUOracle, 
    plot_docking_results, 
    ReceptorPreparer
)
from pathlib import Path
import polars as pl
import subprocess

def main():
    # 1. Configuration
    target_id = "CHEMBL2508" # CDK2
    doc_id = "CHEMBL5728430"
    pdb_id = "5L2S"
    ligand_resname = "6ZV" # The ligand to use for grid box generation. The default is the first hetatm in the PDB file.
    activity_units = "nM" # Units for ChEMBL activity data ('nM', 'uM', etc.)
    
    work_dir = Path("CDK6_workflow")
    work_dir.mkdir(exist_ok=True)
    
    print(f"--- Starting Workflow ({pdb_id}) ---")
    
    # 2. Receptor Preparation
    # Note: module load autogrid autodock-gpu should be done in the environment
    preparer = ReceptorPreparer()
    
    print(f"Preparing receptor from PDB {pdb_id} (Ligand: {ligand_resname})...")
    try:
        # Prepare receptor and grid using mk_prepare_receptor.py logic
        # This will save protein/ligand PDBs and generate PDBQT, GPF, and maps.fld in grid/ directory
        fld_path = preparer.prepare_receptor_and_grid(
            pdb_id, 
            chain='A', 
            output_dir=work_dir, 
            allow_bad_res=True,
            ligand_resname=ligand_resname
        )
        print(f"Grid maps generated at: {fld_path}")
        
    except Exception as e:
        print(f"Failed at receptor preparation step: {e}")
        return

    # 3. Data Collection
    print(f"Fetching ChEMBL data for {target_id} (Units: {activity_units})...")
    df = fetch_chembl_data(target_id, doc_id, units=activity_units)
    print(f"Retrieved {len(df)} compounds with {activity_units} units.")
    
    # Save raw data
    df.write_csv(work_dir / f"{target_id}_{doc_id}_raw.csv")

    # 4. Docking
    print("Initializing AutoDock-GPU Oracle...")
    try:
        oracle = AutoDockGPUOracle(
            receptor_file=fld_path, 
            adgpu_executable="adgpu",
            save_dir=work_dir / "docking_output"
        )
        
        # Get unique SMILES for docking to save time
        all_smiles = df.get_column("canonical_smiles").to_list()
        unique_smiles = list(set(all_smiles))
        print(f"Docking {len(unique_smiles)} unique compounds (from {len(all_smiles)} total)...")
        
        unique_scores = oracle.score_batch(unique_smiles)
        
        # Map scores back to all SMILES
        smiles_to_score = dict(zip(unique_smiles, unique_scores))
        scores = [smiles_to_score[s] for s in all_smiles]
        
        # Add scores to dataframe
        df = df.with_columns(pl.Series(name="docking_score", values=scores))
        
        # Save detailed results CSV
        oracle.results_df.write_csv(work_dir / "docking_results_detailed.csv")
        
        # Save best poses SDF
        oracle.save_best_poses_sdf(work_dir / "best_poses.sdf")
        
    except Exception as e:
        print(f"Failed at docking step: {e}")
        return

    # 5. Analysis & Plotting
    print("Generating results plot...")
    plot_path = work_dir / "docking_analysis.png"
    # Ensure we use log activity if available
    if "standard_value" in df.columns:
        plot_docking_results(
            df, 
            score_col="docking_score", 
            activity_col="standard_value", 
            output_path=str(plot_path),
            activity_units=activity_units
        )
    else:
        print("standard_value missing, plotting against index.")
        df = df.with_row_index("index")
        plot_docking_results(df, score_col="docking_score", activity_col="index", output_path=str(plot_path))
    
    print(f"Workflow complete. Results saved in {work_dir}")

if __name__ == "__main__":
    main()
