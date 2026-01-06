from rdkit import Chem
from argparse import ArgumentParser
from pathlib import Path
import yaml
import polars as pl
from docking_benchmark import (
    fetch_chembl_data,
    AutoDockGPUOracle,
    plot_docking_results,
    plot_activity_distribution,
    ReceptorPreparer,
    fetch_ligand_expo_sdf,
    assign_bond_orders_from_template
)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = ArgumentParser(description="Run Docking Workflow from YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Configuration
    target_id = config.get("target_id")
    doc_id = config.get("doc_id")
    pdb_id = config.get("pdb_id")
    chain = config.get("chain", "A")
    ligand_resname = config.get("ligand_resname")
    activity_units = config.get("activity_units", "nM")
    activity_col = config.get("activity_column", "standard_value")

    # Output directory
    work_dir = Path(config.get("output_dir", f"{pdb_id}_workflow"))
    work_dir.mkdir(exist_ok=True, parents=True)

    # Optional local files
    protein_pdb_path = config.get("protein_pdb_path")
    ligand_pdb_path = config.get("ligand_pdb_path")

    # Filtering parameters
    fragment_smiles = config.get("fragment_smiles")
    rmsd_threshold = config.get("rmsd_threshold", 2.0)

    print(f"--- Starting Workflow ({pdb_id}) ---")
    print(f"Output Directory: {work_dir}")

    # 2. Receptor Preparation
    preparer = ReceptorPreparer()

    print(f"Preparing receptor from PDB {pdb_id}...")
    try:
        fld_path = preparer.prepare_receptor_and_grid(
            pdb_id,
            chain=chain,
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
         
         # Attempt bond order correction using Ligand Expo
         if ligand_resname:
            print(f"Attempting to assign bond orders for {ligand_resname} using Ligand Expo template...")
            try:
                 # Fetch template
                 template_sdf = fetch_ligand_expo_sdf(ligand_resname, work_dir)
                 if template_sdf:
                     # Load PDB ligand
                     pdb_mol = Chem.MolFromPDBFile(str(reference_ligand_path), removeHs=False)
                     
                     # Load Template
                     suppl = Chem.SDMolSupplier(str(template_sdf), removeHs=False)
                     template_mol = next(iter(suppl), None)
                     
                     if pdb_mol and template_mol:
                        # Assign Bond Orders
                        corrected_mol = assign_bond_orders_from_template(pdb_mol, template_mol)
                        if corrected_mol:
                            corrected_path = work_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
                            w = Chem.SDWriter(str(corrected_path))
                            w.write(corrected_mol)
                            w.close()
                            print(f"Successfully assigned bond orders. Saved to {corrected_path}")
                            
                            # Update reference ligand path to use the corrected one
                            reference_ligand_path = corrected_path
                        else:
                            print(f"Warning: Bond order assignment failed for {ligand_resname}. RMSD filtering might be inaccurate.")
            except Exception as e:
                print(f"Bond order assignment warning: {e}. Proceeding with original PDB ligand.")

    # 3. Data Collection
    if config.get("ligand_csv_path"):
        csv_path = Path(config.get("ligand_csv_path"))
        print(f"Loading local ligand data from {csv_path}...")
        if not csv_path.exists():
            raise FileNotFoundError(f"Ligand CSV not found: {csv_path}")
        df = pl.read_csv(csv_path)
        # Ensure we have standard column names for the rest of the script
        if "canonical_smiles" not in df.columns and "smiles" in df.columns:
            df = df.rename({"smiles": "canonical_smiles"})
        
        # Also handle activity column mapping
        if "standard_value" not in df.columns:
            for col in ["ic50", "pic50", "value", "activity", "Ki"]:
                if col in df.columns:
                    df = df.rename({col: "standard_value"})
                    print(f"Renamed column '{col}' to 'standard_value' for activity analysis.")
                    break
    else:
        print(f"Fetching ChEMBL data for {target_id} (Units: {activity_units})...")
        df = fetch_chembl_data(target_id, doc_id, units=activity_units)
    
    print(f"Retrieved/Loaded {len(df)} compounds.")

    # Save raw data
    df.write_csv(work_dir / f"{target_id}_{doc_id}_raw.csv")

    # Plot activity distribution
    print("Generating activity distribution plot...")
    dist_path = work_dir / "activity_distribution.png"
    plot_activity_distribution(
        df,
        activity_col="standard_value" if "standard_value" in df.columns else activity_col,
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
            fragment_smiles=fragment_smiles,
            rmsd_threshold=rmsd_threshold
        )

        # Get unique SMILES for docking to save time
        all_smiles = df.get_column("canonical_smiles").to_list()
        unique_smiles = list(set(all_smiles))
        
        # New Strict Rule: Filter before docking and fail if empty.
        # But Oracle does filtering too. We let Oracle handle strict failure for fragment mismatch.
        # We pass just unique smiles to save compute.
        print(f"Docking {len(unique_smiles)} unique compounds (from {len(all_smiles)} total)...")

        # Run docking (which returns dict {smiles: score})
        # The oracle will raise ValueError if no compounds match the fragment
        smile_to_score = oracle.score_batch(unique_smiles)

        results_df = oracle.results_df

        # Report Statistics
        total = len(results_df)
        valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))

        # Check Fragment matches in input
        fragment_matched = len(results_df.filter(pl.col("fragment_precheck") == True))

        print("\n--- Statistics ---")
        print(f"Total Compounds Processed: {total}")
        if fragment_smiles:
            print(f"Compounds matching Fragment (2D): {fragment_matched} ({fragment_matched/total*100:.1f}%)")
        print(f"Compounds with Valid Poses (RMSD < {rmsd_threshold}): {valid_poses} ({valid_poses/total*100:.1f}%)")

        # Merge scores back to original DF
        # We need to map smile -> score
        # smile_to_score is already {smile: score}
        
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
    
    # Determine which activity column to use for plotting
    plot_act_col = "standard_value" if "standard_value" in df.columns else activity_col
    
    if plot_act_col in df.columns:
        plot_docking_results(
            df,
            score_col="docking_score",
            activity_col=plot_act_col,
            output_path=str(plot_path),
            activity_units=activity_units
        )
    else:
        print(f"Activity column '{plot_act_col}' missing, plotting against index.")
        df = df.with_row_index("index")
        plot_docking_results(df, score_col="docking_score", activity_col="index", output_path=str(plot_path))

    print(f"Workflow complete. Results saved in {work_dir}")

if __name__ == "__main__":
    main()
