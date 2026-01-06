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

    print(f"--- Starting Workflow ({pdb_id}) ---")
    print(f"Output Directory: {work_dir}")
    print(f"Stage: {args.stage}")

    # STAGE 1: Retrieval (Data & Receptor PDB)
    # Note: Receptor PDB fetching is handled in grid stage usually, but ChEMBL is here.
    if args.stage in ["all", "retrieve"]:
        print("\n=== Stage 1: Data Retrieval ===")
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
        raw_csv_path = work_dir / f"{target_id}_{doc_id}_raw.csv"
        df.write_csv(raw_csv_path)
        print(f"Raw data saved to {raw_csv_path}")

        # Plot activity distribution
        print("Generating activity distribution plot...")
        dist_path = work_dir / "activity_distribution.png"
        plot_activity_distribution(
            df,
            activity_col="standard_value" if "standard_value" in df.columns else activity_col,
            output_path=str(dist_path),
            activity_units=activity_units
        )

    # STAGE 2: Grid Preparation
    fld_path = None # Will determine path
    reference_ligand_path = None

    if args.stage in ["all", "grid"]:
        print("\n=== Stage 2: Receptor & Grid Preparation ===")
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
            # Save the path to a file for next stages if needed?
            # For now, we assume standard naming conventions or persistent objects if running 'all'.

        except Exception as e:
            print(f"Failed at receptor preparation step: {e}")
            sys.exit(1)

        # Determine reference ligand path for RMSD calculation
        if ligand_pdb_path:
             reference_ligand_path = work_dir / "grid" / Path(ligand_pdb_path).name
        else:
             reference_ligand_path = work_dir / "grid" / f"{pdb_id}_ligand.pdb"

             # Attempt bond order correction using Ligand Expo
             if ligand_resname:
                print(f"Attempting to assign bond orders for {ligand_resname} using Ligand Expo template...")
                template_sdf_path = None
                try:
                     # Fetch template
                     template_sdf_path = fetch_ligand_expo_sdf(ligand_resname, work_dir)
                     if template_sdf_path:
                         # Load PDB ligand
                         pdb_mol = Chem.MolFromPDBFile(str(reference_ligand_path), removeHs=False)

                         # Load Template
                         suppl = Chem.SDMolSupplier(str(template_sdf_path), removeHs=False)
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
                finally:
                    # Cleanup template
                    if template_sdf_path and template_sdf_path.exists():
                        template_sdf_path.unlink()

    # Recovery of paths if skipping stages
    if args.stage not in ["all", "grid"]:
        # Assume files exist in work_dir
        # Try to find fld
        possible_flds = list((work_dir / "grid").glob("*.maps.fld"))
        if possible_flds:
            fld_path = possible_flds[0]
            print(f"Found existing grid map: {fld_path}")
        
        # Try to find reference ligand
        # Check corrected first
        corrected = work_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
        if corrected.exists():
             reference_ligand_path = corrected
        else:
             reference_ligand_path = work_dir / "grid" / f"{pdb_id}_ligand.pdb"

        if reference_ligand_path.exists():
             print(f"Found reference ligand: {reference_ligand_path}")


    # STAGE 3: Docking
    if args.stage in ["all", "docking"]:
        print("\n=== Stage 3: Docking ===")

        # Load data if not in memory (from stage 1)
        raw_csv_path = work_dir / f"{target_id}_{doc_id}_raw.csv"
        if not raw_csv_path.exists():
            print(f"Error: Raw data file {raw_csv_path} not found. Run 'retrieve' stage first.")
            sys.exit(1)

        df = pl.read_csv(raw_csv_path)

        if not fld_path or not fld_path.exists():
            print("Error: Grid maps not found. Run 'grid' stage first.")
            sys.exit(1)

        print("Initializing AutoDock-GPU Oracle...")
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

            # Get unique SMILES for docking to save time
            all_smiles = df.get_column("canonical_smiles").to_list()
            unique_smiles = list(set(all_smiles))

            print(f"Docking {len(unique_smiles)} unique compounds (from {len(all_smiles)} total)...")

            # Run docking (which returns dict {smiles: score})
            smile_to_score = oracle.score_batch(unique_smiles)

            results_df = oracle.results_df

            # Report Statistics
            total = len(results_df)
            valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))
            fragment_matched = len(results_df.filter(pl.col("fragment_precheck") == True))

            print("\n--- Statistics ---")
            print(f"Total Compounds Processed: {total}")
            if fragment_smiles:
                print(f"Compounds matching Fragment (2D): {fragment_matched} ({fragment_matched/total*100:.1f}%)")
            print(f"Compounds with Valid Poses (RMSD < {rmsd_threshold}): {valid_poses} ({valid_poses/total*100:.1f}%)")

            # Merge results back to original DF
            # We want to carry over validity and "best any score" for plotting

            # Create a dictionary for mapping all results
            # We need valid_pose_found, docking_score, best_any_score

            # Convert results_df to pandas or dictionary for efficient mapping?
            # Polars join is better.

            # Ensure unique results (smiles is unique in results_df because score_batch uniques input)
            # Join df with results_df on 'canonical_smiles' == 'smiles'

            results_df_renamed = results_df.rename({"smiles": "canonical_smiles"})

            # Select relevant columns from results
            # 'docking_score' (valid score), 'best_any_score', 'valid_pose_found'
            # Note: results_df might not have 'best_any_score' if I didn't update score_batch to return it in DF, but I did update _process_chunk.
            # Wait, results_df is created from all_results list of dicts.
            # _process_chunk returns dicts with 'best_any_score'.

            df = df.join(results_df_renamed.select(["canonical_smiles", "docking_score", "best_any_score", "valid_pose_found"]),
                         on="canonical_smiles", how="left")

            # Create a 'plotting_score' column: docking_score if valid, else best_any_score
            # Actually, the user wants 'red dots' for invalid.
            # So we can keep them separate or use a single score column and use valid_pose_found to color.
            # I'll fill docking_score with best_any_score where docking_score is null?
            # No, 'docking_score' in results_df is best_valid_score (or NaN).
            # 'best_any_score' is best score regardless.
            # So I will create a new column `final_score` = coalese(docking_score, best_any_score).

            df = df.with_columns(
                pl.coalesce([pl.col("docking_score"), pl.col("best_any_score")]).alias("final_score")
            )

            # Save detailed results CSV (merged)
            detailed_path = work_dir / "docking_results_detailed.csv"
            df.write_csv(detailed_path)
            print(f"Detailed results saved to {detailed_path}")

            # Save best poses SDF
            oracle.save_best_poses_sdf(work_dir / "best_poses.sdf")

        except Exception as e:
            print(f"Failed at docking step: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


    # STAGE 4: Analysis
    if args.stage in ["all", "analysis", "docking"]: # docking usually triggers analysis
        print("\n=== Stage 4: Analysis ===")

        # Load detailed results if we jumped here
        detailed_path = work_dir / "docking_results_detailed.csv"
        if args.stage == "analysis":
            if not detailed_path.exists():
                print(f"Error: Detailed results {detailed_path} not found. Run 'docking' stage first.")
                sys.exit(1)
            df = pl.read_csv(detailed_path)

        print("Generating results plot...")
        plot_path = work_dir / "docking_analysis.png"

        # Determine which activity column to use for plotting
        plot_act_col = "standard_value" if "standard_value" in df.columns else activity_col

        # Use "final_score" for plotting (contains both valid and invalid scores)
        # Use "valid_pose_found" for coloring

        if "final_score" not in df.columns:
            # Fallback for backward compatibility or if previous step failed
            if "docking_score" in df.columns:
                print("Warning: 'final_score' column missing, using 'docking_score'.")
                df = df.with_columns(pl.col("docking_score").alias("final_score"))
            else:
                print("Error: No score column found for plotting.")
                sys.exit(1)

        if plot_act_col in df.columns:
            plot_docking_results(
                df,
                score_col="final_score",
                activity_col=plot_act_col,
                valid_col="valid_pose_found",
                output_path=str(plot_path),
                activity_units=activity_units
            )
        else:
            print(f"Activity column '{plot_act_col}' missing, plotting against index.")
            df = df.with_row_index("index")
            plot_docking_results(df, score_col="final_score", activity_col="index", valid_col="valid_pose_found", output_path=str(plot_path))

    print(f"\nWorkflow complete. Results saved in {work_dir}")

if __name__ == "__main__":
    main()
