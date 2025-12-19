import argparse
import yaml
from pathlib import Path
import polars as pl
import math
from docking_benchmark import (
    fetch_chembl_data,
    AutoDockGPUOracle,
    plot_docking_results,
    plot_activity_distribution,
    ReceptorPreparer
)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Docking Workflow from YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Configuration
    target_id = config.get("target_id")
    doc_id = config.get("doc_id")
    pdb_id = config.get("pdb_id")
    ligand_resname = config.get("ligand_resname")
    activity_units = config.get("activity_units", "nM")

    # Output directory
    work_dir = Path(config.get("output_dir", f"{pdb_id}_workflow"))
    work_dir.mkdir(exist_ok=True, parents=True)

    # Optional local files
    protein_pdb_path = config.get("protein_pdb_path")
    ligand_pdb_path = config.get("ligand_pdb_path")

    # Filtering parameters
    smarts = config.get("smarts")
    rmsd_threshold = config.get("rmsd_threshold", 2.0)

    print(f"--- Starting Workflow ({pdb_id}) ---")
    print(f"Output Directory: {work_dir}")

    # 2. Receptor Preparation
    preparer = ReceptorPreparer()

    print(f"Preparing receptor from PDB {pdb_id}...")
    try:
        fld_path = preparer.prepare_receptor_and_grid(
            pdb_id,
            chain='A',
            output_dir=work_dir,
            allow_bad_res=True,
            ligand_resname=ligand_resname,
            protein_pdb_path=protein_pdb_path,
            ligand_pdb_path=ligand_pdb_path
        )
        print(f"Grid maps generated at: {fld_path}")

    except Exception as e:
        print(f"Failed at receptor preparation step: {e}")
        return

    # Determine reference ligand path for RMSD calculation
    # If using local ligand PDB, use that. Otherwise use the one extracted by preparer.
    if ligand_pdb_path:
         reference_ligand_path = work_dir / "grid" / Path(ligand_pdb_path).name
    else:
         reference_ligand_path = work_dir / "grid" / f"{pdb_id}_ligand.pdb"

    # 3. Data Collection
    print(f"Fetching ChEMBL data for {target_id} (Units: {activity_units})...")
    df = fetch_chembl_data(target_id, doc_id, units=activity_units)
    print(f"Retrieved {len(df)} compounds with {activity_units} units.")

    # Save raw data
    df.write_csv(work_dir / f"{target_id}_{doc_id}_raw.csv")

    # Plot activity distribution
    print("Generating activity distribution plot...")
    dist_path = work_dir / "activity_distribution.png"
    plot_activity_distribution(
        df,
        activity_col="standard_value",
        output_path=str(dist_path),
        activity_units=activity_units
    )

    # 4. Docking
    print("Initializing AutoDock-GPU Oracle...")
    try:
        oracle = AutoDockGPUOracle(
            receptor_file=fld_path,
            adgpu_executable="adgpu",
            save_dir=work_dir / "docking_output",
            reference_ligand_path=reference_ligand_path,
            smarts=smarts,
            rmsd_threshold=rmsd_threshold
        )

        # Get unique SMILES for docking to save time
        all_smiles = df.get_column("canonical_smiles").to_list()
        unique_smiles = list(set(all_smiles))
        print(f"Docking {len(unique_smiles)} unique compounds (from {len(all_smiles)} total)...")

        # Run docking (which returns scores)
        # We need the full results dataframe to get detailed stats
        unique_scores = oracle.score_batch(unique_smiles)

        results_df = oracle.results_df

        # Report Statistics
        total = len(results_df)
        valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))

        # Check SMARTS matches in input
        smarts_matched = len(results_df.filter(pl.col("smarts_precheck") == True))

        print("\n--- Statistics ---")
        print(f"Total Compounds Processed: {total}")
        if smarts:
            print(f"Compounds matching SMARTS constraint (2D): {smarts_matched} ({smarts_matched/total*100:.1f}%)")
        print(f"Compounds with Valid Poses (RMSD < {rmsd_threshold}): {valid_poses} ({valid_poses/total*100:.1f}%)")

        # Merge scores back to original DF
        # We need to map smile -> score
        # Since unique_smiles was used, we map back.
        smile_to_score = dict(zip(results_df.get_column("smiles").to_list(), results_df.get_column("docking_score").to_list()))

        scores = [smile_to_score.get(s, float('nan')) for s in all_smiles]

        # Add scores to dataframe
        df = df.with_columns(pl.Series(name="docking_score", values=scores))

        # Save detailed results CSV
        # Also map validation info back to detailed CSV
        detailed_path = work_dir / "docking_results_detailed.csv"
        results_df.write_csv(detailed_path)
        print(f"Detailed results saved to {detailed_path}")

        # Save best poses SDF
        oracle.save_best_poses_sdf(work_dir / "best_poses.sdf")

    except Exception as e:
        print(f"Failed at docking step: {e}")
        # traceback
        import traceback
        traceback.print_exc()
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
