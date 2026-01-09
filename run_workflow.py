from rdkit import Chem
from argparse import ArgumentParser
from pathlib import Path
import yaml
import polars as pl
import os
import sys
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

    # Staging arguments
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "retrieve", "grid", "docking", "analysis"],
                        help="Run a specific stage of the workflow. 'all' runs everything.")

    # User-friendly robustness arguments
    parser.add_argument("--smarts", type=str, help="Override SMARTS string for fragment filtering.")
    parser.add_argument("--no_isomers", action="store_true", help="Disable stereoisomer generation (respect input stereo).")

    args = parser.parse_args()

    config = load_config(args.config)

    # Override config with CLI args
    if args.smarts:
        config["fragment_smiles"] = args.smarts
        print(f"Overriding fragment SMARTS with: {args.smarts}")

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

    # Determine naming prefix for output files
    if target_id and doc_id:
        data_prefix = f"{target_id}_{doc_id}"
    elif target_id:
        data_prefix = f"{target_id}"
    elif config.get("ligand_csv_path"):
        data_prefix = Path(config["ligand_csv_path"]).stem
    else:
        data_prefix = "workflow_data"

    print(f"\n" + "="*50)
    print(f"--- Starting Docking Workflow: {pdb_id} ---")
    print(f"Output Directory: {work_dir}")
    print(f"Workflow Prefix: {data_prefix}")
    print(f"Stage: {args.stage}")
    print("="*50 + "\n")

    # STAGE 1: Retrieval (Data & Receptor PDB)
    if args.stage in ["all", "retrieve"]:
        print(f"[STAGE 1] Data Retrieval")
        print(f"------------------------")
        if config.get("ligand_csv_path"):
            csv_path = Path(config.get("ligand_csv_path"))
            print(f"-> Loading local ligand data from: {csv_path}")
            if not csv_path.exists():
                raise FileNotFoundError(f"Ligand CSV not found: {csv_path}")
            df = pl.read_csv(csv_path)
            
            # 1. Standardize SMILES column
            # If canonical_smiles is missing, look for common names
            if "canonical_smiles" not in df.columns:
                smi_fallbacks = ["smiles", "SMILES", "SMILE", "canonical_smiles"]
                for col in smi_fallbacks:
                    if col in df.columns:
                        df = df.rename({col: "canonical_smiles"})
                        print(f"   Note: Using '{col}' as the SMILES column.")
                        break

            # 2. Standardize Activity column
            # Priority: 1. YAML specified activity_column, 2. "standard_value", 3. Common fallbacks
            if activity_col in df.columns:
                if activity_col != "standard_value":
                    df = df.rename({activity_col: "standard_value"})
                    print(f"   Note: Using '{activity_col}' as the activity column.")
            elif "standard_value" not in df.columns:
                act_fallbacks = ["ic50", "pic50", "value", "activity", "Ki", "IC50", "KI"]
                for col in act_fallbacks:
                    if col in df.columns:
                        df = df.rename({col: "standard_value"})
                        print(f"   Note: Identified '{col}' as activity column and mapped to 'standard_value'.")
                        break
        else:
            print(f"-> Fetching ChEMBL data for {target_id} (Units: {activity_units})...")
            df = fetch_chembl_data(target_id, doc_id, units=activity_units)

        # Verify we have the required columns
        if "canonical_smiles" not in df.columns:
            print("ERROR: Could not find SMILES column in data. Please check CSV or YAML.")
            sys.exit(1)
        if "standard_value" not in df.columns:
             print("WARNING: Could not identify activity column. Analysis will be limited.")

        print(f"-> Successfully loaded {len(df)} compounds.")

        # Save raw data
        raw_csv_path = work_dir / f"{data_prefix}_cleaned_data.csv"
        df.write_csv(raw_csv_path)
        print(f"-> Standardized data saved to: {raw_csv_path}")

        # Plot activity distribution
        # Check if we have pchembl_value (preferred) or standard_value
        if "pchembl_value" in df.columns:
            print("-> Generating activity distribution plot (using pchembl_value)...")
            dist_path = work_dir / f"{data_prefix}_activity_dist.png"
            plot_activity_distribution(
                df,
                activity_col="pchembl_value",  # Use pchembl_value directly (unit-agnostic)
                output_path=str(dist_path),
                activity_units=None  # pchembl is unit-agnostic
            )
            print(f"   Saved to: {dist_path}")
        elif "standard_value" in df.columns:
            print("-> Generating activity distribution plot...")
            dist_path = work_dir / f"{data_prefix}_activity_dist.png"
            plot_activity_distribution(
                df,
                activity_col="standard_value",
                output_path=str(dist_path),
                activity_units=activity_units  # Required for standard_value
            )
            print(f"   Saved to: {dist_path}")

    # STAGE 2: Grid Preparation
    fld_path = None 
    reference_ligand_path = None

    if args.stage in ["all", "grid"]:
        print(f"\n[STAGE 2] Receptor & Grid Preparation")
        print(f"-------------------------------------")
        preparer = ReceptorPreparer()

        print(f"-> Preparing receptor from PDB: {pdb_id} (Chain: {chain})")
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
            print(f"-> Grid maps generated at: {fld_path}")

        except Exception as e:
            print(f"ERROR during receptor preparation: {e}")
            sys.exit(1)

        # Determine reference ligand path for RMSD calculation
        if ligand_pdb_path:
             reference_ligand_path = work_dir / "grid" / Path(ligand_pdb_path).name
        else:
             reference_ligand_path = work_dir / "grid" / f"{pdb_id}_ligand.pdb"

             if ligand_resname:
                print(f"-> Attempting to assign bond orders for '{ligand_resname}' using Ligand Expo template...")
                template_sdf_path = None
                try:
                     template_sdf_path = fetch_ligand_expo_sdf(ligand_resname, work_dir)
                     if template_sdf_path:
                         pdb_mol = Chem.MolFromPDBFile(str(reference_ligand_path), removeHs=False)
                         suppl = Chem.SDMolSupplier(str(template_sdf_path), removeHs=False)
                         template_mol = next(iter(suppl), None)

                         if pdb_mol and template_mol:
                            corrected_mol = assign_bond_orders_from_template(pdb_mol, template_mol)
                            if corrected_mol:
                                corrected_path = work_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
                                w = Chem.SDWriter(str(corrected_path))
                                w.write(corrected_mol)
                                w.close()
                                print(f"   Success! Saved corrected ligand to {corrected_path}")
                                reference_ligand_path = corrected_path
                            else:
                                print(f"   Warning: Bond order assignment failed.")
                except Exception as e:
                    print(f"   Warning: Bond order correction encountered an error: {e}")
                finally:
                    if template_sdf_path and template_sdf_path.exists():
                        template_sdf_path.unlink()

    # Recovery of paths if skipping stages
    if args.stage not in ["all", "grid"]:
        possible_flds = list((work_dir / "grid").glob("*.maps.fld"))
        if possible_flds:
            fld_path = possible_flds[0]
            print(f"-> Found existing grid map: {fld_path}")
        
        corrected = work_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
        if corrected.exists():
             reference_ligand_path = corrected
        else:
             reference_ligand_path = work_dir / "grid" / f"{pdb_id}_ligand.pdb"

        if reference_ligand_path and reference_ligand_path.exists():
             print(f"-> Found reference ligand: {reference_ligand_path}")

    # STAGE 3: Docking
    if args.stage in ["all", "docking"]:
        print(f"\n[STAGE 3] Docking Execution")
        print(f"--------------------------")

        # Load data if not in memory (from stage 1)
        raw_csv_path = work_dir / f"{data_prefix}_cleaned_data.csv"
        if not raw_csv_path.exists():
            print(f"ERROR: Data file {raw_csv_path} not found. Please run 'retrieve' stage first.")
            sys.exit(1)

        df = pl.read_csv(raw_csv_path)

        if not fld_path or not fld_path.exists():
            print("ERROR: Grid maps not found. Please run 'grid' stage first.")
            sys.exit(1)

        print("-> Initializing AutoDock-GPU Oracle...")
        try:
            oracle = AutoDockGPUOracle(
                receptor_file=fld_path,
                adgpu_executable="adgpu",
                save_dir=work_dir / "docking_output",
                reference_ligand_path=reference_ligand_path,
                fragment_smiles=fragment_smiles,
                rmsd_threshold=rmsd_threshold,
                generate_isomers=not args.no_isomers
            )

            all_smiles = df.get_column("canonical_smiles").to_list()
            unique_smiles = list(set(all_smiles))

            print(f"-> Docking {len(unique_smiles)} unique compounds...")
            smile_to_score = oracle.score_batch(unique_smiles)

            results_df = oracle.results_df
            total = len(results_df)
            valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))
            fragment_matched = len(results_df.filter(pl.col("fragment_precheck") == True))
            total_conformers = results_df["n_conformers"].sum()

            print("\nDocking Statistics:")
            print(f"  - Total Unique Compounds: {total}")
            print(f"  - Total Conformers Docked: {total_conformers}")
            if fragment_smiles:
                print(f"  - Matches Fragment (2D): {fragment_matched} ({fragment_matched/total*100:.1f}%)")
            print(f"  - Valid Poses (RMSD < {rmsd_threshold}): {valid_poses} ({valid_poses/total*100:.1f}%)")

            # Merge results
            results_df_renamed = results_df.rename({"smiles": "canonical_smiles"})
            
            # Select relevant columns from results. 
            df = df.join(results_df_renamed.select(["canonical_smiles", "docking_score", "score_valid", "score_best_any", "dlg_path_valid", "dlg_path_any", "valid_pose_found", "n_conformers"]),
                         on="canonical_smiles", how="left")

            # Fill nulls in validity column for plotting consistency
            df = df.with_columns(
                pl.col("valid_pose_found").fill_null(False)
            )

            # Save results
            detailed_path = work_dir / f"{data_prefix}_vs_{pdb_id}_results_full.csv"
            df.write_csv(detailed_path)
            print(f"\n-> Full docking results saved to: {detailed_path}")

            # 1. Save RMSD-constrained poses (with fallback to best any if none valid)
            poses_valid_path = work_dir / f"{data_prefix}_vs_{pdb_id}_poses_rmsd_constrained.sdf"
            oracle.save_best_poses_sdf(
                output_path=poses_valid_path, 
                df_metadata=df, 
                id_col=config.get("id_column", "id"),
                score_col="docking_score", # This already has the fallback logic (valid if exists, else best any)
                dlg_col="dlg_path"
            )

            # 2. Save Absolute Best poses (regardless of RMSD)
            poses_any_path = work_dir / f"{data_prefix}_vs_{pdb_id}_poses_best_any.sdf"
            oracle.save_best_poses_sdf(
                output_path=poses_any_path, 
                df_metadata=df, 
                id_col=config.get("id_column", "id"),
                score_col="score_best_any",
                dlg_col="dlg_path_any"
            )

        except Exception as e:
            print(f"FATAL ERROR during docking step: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # STAGE 4: Analysis
    if args.stage in ["all", "analysis", "docking"]:
        print(f"\n[STAGE 4] Analysis & Plotting")
        print(f"----------------------------")

        detailed_path = work_dir / f"{data_prefix}_vs_{pdb_id}_results_full.csv"
        if args.stage == "analysis":
            if not detailed_path.exists():
                print(f"ERROR: Detailed results {detailed_path} not found. Run 'docking' stage first.")
                sys.exit(1)
            df = pl.read_csv(detailed_path)

        # Determine which activity column to use for plotting
        # If pchembl_value exists, standard_value will also exist (created from pchembl)
        # Use standard_value for consistency, but pass None for units if pchembl exists
        if "pchembl_value" in df.columns:
            plot_act_col = "standard_value"  # Created from pchembl, already in pActivity units
        elif "standard_value" in df.columns:
            plot_act_col = "standard_value"
        else:
            plot_act_col = activity_col

        if "score_best_any" not in df.columns:
            print("ERROR: Score columns missing for plotting. Check docking stage.")
            sys.exit(1)

        # 1. Plot Unconstrained (Best Any)
        plot_any_path = work_dir / f"{data_prefix}_vs_{pdb_id}_analysis_best_any.png"
        if plot_act_col in df.columns:
            print(f"-> Generating Unconstrained Activity vs Score plot...")
            # Use None for activity_units if we have pchembl_value (unit-agnostic)
            plot_units = None if "pchembl_value" in df.columns else activity_units
            plot_docking_results(
                df,
                score_col="score_best_any",
                activity_col=plot_act_col,
                valid_col="valid_pose_found",
                output_path=str(plot_any_path),
                activity_units=plot_units
            )
        
        # 2. Plot RMSD-constrained (with fallback)
        plot_valid_path = work_dir / f"{data_prefix}_vs_{pdb_id}_analysis_rmsd_constrained.png"
        if plot_act_col in df.columns:
            print(f"-> Generating RMSD-Constrained Activity vs Score plot...")
            # Use None for activity_units if we have pchembl_value (unit-agnostic)
            plot_units = None if "pchembl_value" in df.columns else activity_units
            plot_docking_results(
                df,
                score_col="docking_score", # Use the fallback-enabled score
                activity_col=plot_act_col,
                valid_col="valid_pose_found",
                output_path=str(plot_valid_path),
                activity_units=plot_units
            )

    print(f"\n" + "="*50)
    print(f"Workflow Complete. All results available in: {work_dir}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
