from rdkit import Chem
from argparse import ArgumentParser
from pathlib import Path
import yaml
import polars as pl
import os
import sys
import time
import traceback
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from docking_benchmark import (
    fetch_chembl_data,
    AutoDockGPUOracle,
    plot_docking_results,
    plot_activity_distribution,
    ReceptorPreparer,
    fetch_ligand_expo_sdf,
    assign_bond_orders_from_template
)
from docking_benchmark.receptor import PDBDownloadError, LigandNotFoundError, GridPrepError

class WorkflowStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED_RETRIEVAL = "FAILED_RETRIEVAL"
    FAILED_PDB_404 = "FAILED_PDB_404"
    FAILED_LIGAND_MISSING = "FAILED_LIGAND_MISSING"
    FAILED_GRID_PREP = "FAILED_GRID_PREP"
    FAILED_REF_MATCH = "FAILED_REF_MATCH"
    FAILED_DOCKING = "FAILED_DOCKING"
    FAILED_ANALYSIS = "FAILED_ANALYSIS"

@dataclass
class WorkflowResult:
    config_file: str
    target_id: str
    pdb_id: str
    doc_id: str
    fragment_smiles: str
    status: str
    error_message: str = ""
    n_compounds_total: int = 0
    n_compounds_standardized: int = 0
    n_compounds_matched_2d: int = 0
    n_compounds_docked: int = 0
    n_conformers_docked: int = 0
    n_valid_poses: int = 0
    runtime_seconds: float = 0.0

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
    parser.add_argument("--run-dir", type=str, help="Base directory for the benchmark run (organized outputs)")

    args = parser.parse_args()
    start_time = time.time()

    config = load_config(args.config)

    # 1. Configuration
    target_id = config.get("target_id")
    doc_id = config.get("doc_id")
    pdb_id = config.get("pdb_id")
    ligand_resname = config.get("ligand_resname")
    activity_units = config.get("activity_units", "nM")
    activity_col = config.get("activity_column", "standard_value")

    # Output directory organization
    target_pdb_name = f"{target_id}_{pdb_id}"
    if args.run_dir:
        run_base = Path(args.run_dir)
        run_base.mkdir(exist_ok=True, parents=True)
        # Organize as:
        #   <run_dir>/<target_id>_<pdb_id>/grid/...            (shared per receptor/PDB)
        #   <run_dir>/<target_id>_<pdb_id>/<doc_id>/...       (doc-specific outputs)
        target_pdb_dir = run_base / target_pdb_name
        target_pdb_dir.mkdir(exist_ok=True, parents=True)

        # Doc-specific directory prevents overwrites when multiple docs share same target+PDB
        if doc_id:
            work_dir = target_pdb_dir / str(doc_id)
        else:
            work_dir = target_pdb_dir

        # Grid (receptor maps) should be shared across docs for same PDB
        grid_base_dir = target_pdb_dir
        # Create debug directory in run-dir
        debug_dir = run_base / "debug"
        debug_dir.mkdir(exist_ok=True)
    else:
        work_dir = Path(config.get("output_dir", f"{target_pdb_name}_workflow"))
        debug_dir = None
        grid_base_dir = work_dir

    work_dir.mkdir(exist_ok=True, parents=True)

    # Optional local files
    protein_pdb_path = config.get("protein_pdb_path")
    ligand_pdb_path = config.get("ligand_pdb_path")

    # Filtering parameters
    fragment_smiles = config.get("fragment_smiles")
    rmsd_threshold = config.get("rmsd_threshold", 2.0)

    # Result tracking
    result = WorkflowResult(
        config_file=args.config,
        target_id=target_id,
        pdb_id=pdb_id,
        doc_id=doc_id,
        fragment_smiles=fragment_smiles or "",
        status=WorkflowStatus.SUCCESS.value
    )

    def save_debug_info(error_type, pdb_id, message, extra_files=None):
        if not debug_dir:
            return
        type_dir = debug_dir / error_type
        type_dir.mkdir(exist_ok=True)
        
        info_path = type_dir / f"{pdb_id}_error.txt"
        with open(info_path, "w") as f:
            f.write(f"PDB: {pdb_id}\n")
            f.write(f"Error: {message}\n")
            f.write("-" * 20 + "\n")
            f.write(traceback.format_exc())
            
        if extra_files:
            import shutil
            for ef in extra_files:
                if Path(ef).exists():
                    shutil.copy2(ef, type_dir / f"{pdb_id}_{Path(ef).name}")

    def append_to_summary():
        if not args.run_dir:
            return
        summary_path = Path(args.run_dir) / "benchmark_summary.csv"
        result.runtime_seconds = time.time() - start_time
        
        row = asdict(result)
        df_row = pl.DataFrame([row])
        
        # Use file locking/append mode if possible. 
        # For simplicity with polars, we check if exists
        header = not summary_path.exists()
        with open(summary_path, "a") as f:
            df_row.write_csv(f, include_header=header)

    # Determine naming prefix for output files
    # Prefer a fully qualified prefix so filenames are self-explanatory even outside folder context.
    if target_id and pdb_id and doc_id:
        data_prefix = f"{target_id}_{pdb_id}_{doc_id}"
    elif target_id and pdb_id:
        data_prefix = f"{target_id}_{pdb_id}"
    elif target_id and doc_id:
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
        try:
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
                # Capture printed info for standardization stats
                df, stats = fetch_chembl_data(target_id, doc_id, units=activity_units, return_stats=True)
                result.n_compounds_total = stats.get("n_total", 0)
                result.n_compounds_standardized = stats.get("n_standardized", 0)

            if df is None or df.is_empty():
                print("ERROR: No data retrieved.")
                result.status = WorkflowStatus.FAILED_RETRIEVAL.value
                result.error_message = "No data retrieved from ChEMBL or CSV"
                append_to_summary()
                sys.exit(1)
            
            # For local CSV, we set total/standardized here if not already set
            if result.n_compounds_total == 0:
                result.n_compounds_total = len(df)
                result.n_compounds_standardized = len(df)
        except Exception as e:
            print(f"ERROR during data retrieval: {e}")
            result.status = WorkflowStatus.FAILED_RETRIEVAL.value
            result.error_message = str(e)
            save_debug_info("FAILED_RETRIEVAL", pdb_id, str(e))
            append_to_summary()
            sys.exit(1)

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

        print(f"-> Preparing receptor from PDB: {pdb_id}")
        try:
            fld_path = preparer.prepare_receptor_and_grid(
                pdb_id,
                output_dir=grid_base_dir,
                allow_bad_res=True,
                ligand_resname=ligand_resname,
                protein_pdb_path=protein_pdb_path,
                ligand_pdb_path=ligand_pdb_path
            )
            print(f"-> Grid maps generated at: {fld_path}")

        except PDBDownloadError as e:
            err_msg = str(e)
            print(f"ERROR: PDB download failed: {err_msg}")
            result.status = WorkflowStatus.FAILED_PDB_404.value
            result.error_message = err_msg
            save_debug_info("FAILED_PDB_404", pdb_id, err_msg)
            append_to_summary()
            sys.exit(1)

        except LigandNotFoundError as e:
            err_msg = str(e)
            print(f"ERROR: Ligand not found: {err_msg}")
            result.status = WorkflowStatus.FAILED_LIGAND_MISSING.value
            result.error_message = err_msg
            save_debug_info("FAILED_LIGAND_MISSING", pdb_id, err_msg)
            append_to_summary()
            sys.exit(1)

        except GridPrepError as e:
            err_msg = str(e)
            print(f"ERROR during grid preparation: {err_msg}")
            result.status = WorkflowStatus.FAILED_GRID_PREP.value
            result.error_message = err_msg
            glg_files = list((work_dir / "grid").glob("*.glg"))
            save_debug_info("FAILED_GRID_PREP", pdb_id, err_msg, extra_files=glg_files)
            append_to_summary()
            sys.exit(1)

        except Exception as e:
            err_msg = str(e)
            print(f"ERROR during receptor preparation: {err_msg}")
            result.status = WorkflowStatus.FAILED_GRID_PREP.value
            result.error_message = err_msg
            append_to_summary()
            sys.exit(1)

        # Determine reference ligand path for RMSD calculation
        if ligand_pdb_path:
             reference_ligand_path = grid_base_dir / "grid" / Path(ligand_pdb_path).name
        else:
             reference_ligand_path = grid_base_dir / "grid" / f"{pdb_id}_ligand.pdb"

             if ligand_resname:
                print(f"-> Attempting to assign bond orders for '{ligand_resname}' using Ligand Expo template...")
                template_sdf_path = None
                try:
                     # Use asyncio.run for the async function
                     template_sdf_path = asyncio.run(fetch_ligand_expo_sdf(ligand_resname, work_dir))
                     if template_sdf_path:
                         pdb_mol = Chem.MolFromPDBFile(str(reference_ligand_path), removeHs=False)
                         suppl = Chem.SDMolSupplier(str(template_sdf_path), removeHs=False)
                         template_mol = next(iter(suppl), None)

                         if pdb_mol and template_mol:
                            corrected_mol = assign_bond_orders_from_template(pdb_mol, template_mol)
                            if corrected_mol:
                                corrected_path = grid_base_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
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
        possible_flds = list((grid_base_dir / "grid").glob("*.maps.fld"))
        if possible_flds:
            fld_path = possible_flds[0]
            print(f"-> Found existing grid map: {fld_path}")
        
        corrected = grid_base_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
        if corrected.exists():
             reference_ligand_path = corrected
        else:
             reference_ligand_path = grid_base_dir / "grid" / f"{pdb_id}_ligand.pdb"

        if reference_ligand_path and reference_ligand_path.exists():
             print(f"-> Found reference ligand: {reference_ligand_path}")

    # Reference Ligand Fragment Validation
    if args.stage in ["all", "docking"] and fragment_smiles and reference_ligand_path:
        print(f"-> Validating reference ligand against fragment pattern: {fragment_smiles}")
        try:
            if str(reference_ligand_path).endswith(".sdf"):
                suppl = Chem.SDMolSupplier(str(reference_ligand_path), removeHs=False)
                ref_mol = next(iter(suppl), None)
            else:
                ref_mol = Chem.MolFromPDBFile(str(reference_ligand_path), removeHs=False)
            
            fragment_mol = Chem.MolFromSmiles(fragment_smiles) or Chem.MolFromSmarts(fragment_smiles)
            
            if ref_mol and fragment_mol:
                # Use a robust matching if possible, but basic HasSubstructMatch is a good start
                if not ref_mol.HasSubstructMatch(fragment_mol):
                    # Try adjusting query properties for robustness (similar to what's in docking.py)
                    params = Chem.AdjustQueryParameters()
                    params.makeBondsGeneric = True
                    params.aromatizeIfPossible = True
                    loose_fragment = Chem.AdjustQueryProperties(fragment_mol, params)
                    
                    if not ref_mol.HasSubstructMatch(loose_fragment):
                        print("FATAL: Crystal ligand does not contain the fragment pattern!")
                        result.status = WorkflowStatus.FAILED_REF_MATCH.value
                        result.error_message = f"Crystal ligand ({reference_ligand_path.name}) missing fragment substructure"
                        save_debug_info("FAILED_REF_MATCH", pdb_id, result.error_message)
                        append_to_summary()
                        sys.exit(0) # Exit gracefully as this is an expected incompatibility
            print("   Reference ligand validation successful.")
        except Exception as e:
            print(f"Warning during reference validation: {e}")

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
            if results_df is None:
                raise RuntimeError("Docking results missing from oracle.")

            # Ensure expected columns exist (older runs may omit them)
            if "skip_reason" not in results_df.columns:
                results_df = results_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("skip_reason"))
            if "dlg_path" not in results_df.columns:
                results_df = results_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("dlg_path"))
            if "dlg_path_valid" not in results_df.columns:
                results_df = results_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("dlg_path_valid"))
            if "dlg_path_any" not in results_df.columns:
                results_df = results_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("dlg_path_any"))
            if "n_conformers" not in results_df.columns:
                results_df = results_df.with_columns(pl.lit(0).cast(pl.Int64).alias("n_conformers"))
            total = len(results_df)
            valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))
            fragment_matched = len(results_df.filter(pl.col("fragment_precheck") == True))
            docked_df = results_df.filter(pl.col("skip_reason").is_null())
            total_conformers = int(docked_df["n_conformers"].sum()) if len(docked_df) else 0
            
            # Compounds actually docked are those that were NOT skipped
            # Actually, results_df has a fragment_precheck column
            n_docked = len(docked_df)

            print("\nDocking Statistics:")
            print(f"  - Total Unique Compounds: {total}")
            print(f"  - Total Conformers Docked: {total_conformers}")
            if fragment_smiles:
                print(f"  - Matches Fragment (2D): {fragment_matched} ({fragment_matched/total*100:.1f}%)")
            print(f"  - Valid Poses (RMSD < {rmsd_threshold}): {valid_poses} ({valid_poses/total*100:.1f}%)")

            # Merge results
            results_df_renamed = results_df.rename({"smiles": "canonical_smiles"})
            
            # Select relevant columns from results. 
            # Added skip_reason
            df = df.join(results_df_renamed.select(["canonical_smiles", "docking_score", "score_valid", "score_best_any", "dlg_path", "dlg_path_valid", "dlg_path_any", "valid_pose_found", "n_conformers", "skip_reason"]),
                         on="canonical_smiles", how="left")

            # Fill nulls in validity column for plotting consistency
            df = df.with_columns(
                pl.col("valid_pose_found").fill_null(False)
            )

            # Save results
            detailed_path = work_dir / f"{data_prefix}_results_full.csv"
            df.write_csv(detailed_path)
            print(f"\n-> Full docking results saved to: {detailed_path}")

            # 1. Save RMSD-constrained poses (with fallback to best any if none valid)
            poses_valid_path = work_dir / f"{data_prefix}_poses_rmsd_constrained.sdf"
            oracle.save_best_poses_sdf(
                output_path=poses_valid_path, 
                df_metadata=df, 
                id_col=config.get("id_column", "id"),
                score_col="docking_score", # This already has the fallback logic (valid if exists, else best any)
                dlg_col="dlg_path"
            )

            # 2. Save Absolute Best poses (regardless of RMSD)
            poses_any_path = work_dir / f"{data_prefix}_poses_best_any.sdf"
            oracle.save_best_poses_sdf(
                output_path=poses_any_path, 
                df_metadata=df, 
                id_col=config.get("id_column", "id"),
                score_col="score_best_any",
                dlg_col="dlg_path_any"
            )

            # Update results for summary
            result.n_compounds_matched_2d = fragment_matched
            result.n_compounds_docked = n_docked
            result.n_conformers_docked = total_conformers
            result.n_valid_poses = valid_poses

        except Exception as e:
            print(f"FATAL ERROR during docking step: {e}")
            traceback.print_exc()
            result.status = WorkflowStatus.FAILED_DOCKING.value
            result.error_message = str(e)
            save_debug_info("FAILED_DOCKING", pdb_id, str(e))
            append_to_summary()
            sys.exit(1)

    # STAGE 4: Analysis
    if args.stage in ["all", "analysis", "docking"]:
        print(f"\n[STAGE 4] Analysis & Plotting")
        print(f"----------------------------")

        detailed_path = work_dir / f"{data_prefix}_results_full.csv"
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
        plot_any_path = work_dir / f"{data_prefix}_analysis_best_any.png"
        best_any_metrics = None
        if plot_act_col in df.columns:
            print(f"-> Generating Unconstrained Activity vs Score plot...")
            # Use None for activity_units if we have pchembl_value (unit-agnostic)
            plot_units = None if "pchembl_value" in df.columns else activity_units
            best_any_metrics = plot_docking_results(
                df,
                score_col="score_best_any",
                activity_col=plot_act_col,
                valid_col="valid_pose_found",
                output_path=str(plot_any_path),
                activity_units=plot_units
            )
        
        # 2. Plot RMSD-constrained (with fallback)
        plot_valid_path = work_dir / f"{data_prefix}_analysis_rmsd_constrained.png"
        rmsd_constrained_metrics = None
        if plot_act_col in df.columns:
            print(f"-> Generating RMSD-Constrained Activity vs Score plot...")
            # Use None for activity_units if we have pchembl_value (unit-agnostic)
            plot_units = None if "pchembl_value" in df.columns else activity_units
            rmsd_constrained_metrics = plot_docking_results(
                df,
                score_col="docking_score", # Use the fallback-enabled score
                activity_col=plot_act_col,
                valid_col="valid_pose_found",
                output_path=str(plot_valid_path),
                activity_units=plot_units
            )

        # Save metrics.json (uses the same metrics as plotted)
        try:
            metrics_payload = {
                "target_id": target_id,
                "doc_id": doc_id,
                "pdb_id": pdb_id,
                "fragment_smiles": fragment_smiles,
                "activity_col": plot_act_col,
                "activity_units": (None if "pchembl_value" in df.columns else activity_units),
                "n_compounds_total": result.n_compounds_total,
                "n_compounds_standardized": result.n_compounds_standardized,
                "n_compounds_matched_2d": result.n_compounds_matched_2d,
                "n_compounds_docked": result.n_compounds_docked,
                "n_conformers_docked": result.n_conformers_docked,
                "n_valid_poses": result.n_valid_poses,
                "metrics": {
                    "best_any": best_any_metrics,
                    "rmsd_constrained": rmsd_constrained_metrics,
                },
            }
            metrics_path = work_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        except Exception as e:
            print(f"Warning: failed to write metrics.json: {e}")

    print(f"\n" + "="*50)
    print(f"Workflow Complete. All results available in: {work_dir}")
    print("="*50 + "\n")
    
    append_to_summary()

if __name__ == "__main__":
    main()
