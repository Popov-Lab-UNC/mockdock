# Standard library imports
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import polars as pl
import yaml
from rdkit import Chem, RDLogger

# Local imports
try:
    from .data import fetch_chembl_data
    from .docking import AutoDockGPUOracle
    from .receptor import GridPrepError, LigandNotFoundError, PDBDownloadError, ReceptorPreparer
    from .utils import (
        assign_bond_orders_from_template,
        fetch_ligand_expo_sdf,
        plot_activity_distribution,
        plot_docking_results,
    )
except ImportError:
    # Handle case where it's run as a script or different context
    from fcgmb.data import fetch_chembl_data
    from fcgmb.docking import AutoDockGPUOracle
    from fcgmb.receptor import GridPrepError, LigandNotFoundError, PDBDownloadError, ReceptorPreparer
    from fcgmb.utils import (
        assign_bond_orders_from_template,
        fetch_ligand_expo_sdf,
        plot_activity_distribution,
        plot_docking_results,
    )

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')
# Silence general warnings (like multiprocessing/fork deprecations)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

def load_config(config_path: Union[str, Path]) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_docking_workflow(
    config: dict,
    stage: str = "all",
    smarts: Optional[str] = None,
    no_isomers: bool = False,
    run_dir: Optional[Union[str, Path]] = None,
    config_file_path: Optional[str] = None,
    quiet: bool = False
) -> WorkflowResult:
    """
    Core function to run the docking workflow.
    """
    start_time = time.time()

    # 1. Configuration extraction
    target_id = config.get("target_id")
    doc_id = config.get("doc_id")
    pdb_id = config.get("pdb_id")
    ligand_resname = config.get("ligand_resname")
    activity_units = config.get("activity_units", "nM")
    activity_col = config.get("activity_column", "standard_value")

    # Output directory organization
    target_pdb_name = f"{target_id}_{pdb_id}"
    if run_dir:
        run_base = Path(run_dir)
        run_base.mkdir(exist_ok=True, parents=True)
        target_pdb_dir = run_base / target_pdb_name
        target_pdb_dir.mkdir(exist_ok=True, parents=True)

        if doc_id:
            work_dir = target_pdb_dir / str(doc_id)
        else:
            work_dir = target_pdb_dir

        grid_base_dir = target_pdb_dir
        debug_dir = run_base / "debug"
        debug_dir.mkdir(exist_ok=True)
    else:
        work_dir = Path(config.get("output_dir", f"{target_pdb_name}_workflow"))
        debug_dir = None
        grid_base_dir = work_dir

    work_dir.mkdir(exist_ok=True, parents=True)

    protein_pdb_path = config.get("protein_pdb_path")
    ligand_pdb_path = config.get("ligand_pdb_path")

    fragment_smiles = smarts if smarts else config.get("fragment_smiles")
    rmsd_threshold = config.get("rmsd_threshold", 2.0)

    result = WorkflowResult(
        config_file=config_file_path or "manual",
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
        if not run_dir:
            return
        summary_path = Path(run_dir) / "benchmark_summary.csv"
        result.runtime_seconds = time.time() - start_time
        
        row = asdict(result)
        df_row = pl.DataFrame([row])
        
        header = not summary_path.exists()
        with open(summary_path, "a") as f:
            df_row.write_csv(f, include_header=header)

    if target_id and pdb_id and doc_id:
        data_prefix = f"{target_id}_{pdb_id}_{doc_id}"
    else:
        data_prefix = "workflow_data"

    if not quiet:
        print(f"\n" + "="*50)
        print(f"--- Starting Docking Workflow: {pdb_id} ---")
        print(f"Output Directory: {work_dir}")
        print(f"Workflow Prefix: {data_prefix}")
        print(f"Stage: {stage}")
        print("="*50 + "\n")

    # STAGE 1: Retrieval
    if stage in ["all", "retrieve"]:
        if not quiet: print(f"[STAGE 1] Data Retrieval")
        try:
            if config.get("ligand_csv_path"):
                df = pl.read_csv(Path(config.get("ligand_csv_path")))
                # Basic standardization for CSV
                if "canonical_smiles" not in df.columns:
                     for col in ["smiles", "SMILES", "SMILE"]:
                         if col in df.columns:
                             df = df.rename({col: "canonical_smiles"})
                             break
            else:
                df, stats = fetch_chembl_data(target_id, doc_id, units=activity_units, return_stats=True)
                result.n_compounds_total = stats.get("n_total", 0)
                result.n_compounds_standardized = stats.get("n_standardized", 0)

            if df is None or df.is_empty():
                result.status = WorkflowStatus.FAILED_RETRIEVAL.value
                result.error_message = "No data retrieved"
                append_to_summary()
                return result
            
            if result.n_compounds_total == 0:
                result.n_compounds_total = len(df)
                result.n_compounds_standardized = len(df)
                
            raw_csv_path = work_dir / f"{data_prefix}_cleaned_data.csv"
            df.write_csv(raw_csv_path)
            
            # Plot distribution
            dist_path = work_dir / f"{data_prefix}_activity_dist.png"
            if "pchembl_value" in df.columns:
                plot_activity_distribution(df, activity_col="pchembl_value", output_path=str(dist_path))
            elif "standard_value" in df.columns:
                plot_activity_distribution(df, activity_col="standard_value", output_path=str(dist_path), activity_units=activity_units)
                
        except Exception as e:
            result.status = WorkflowStatus.FAILED_RETRIEVAL.value
            result.error_message = str(e)
            save_debug_info("FAILED_RETRIEVAL", pdb_id, str(e))
            append_to_summary()
            return result

    # STAGE 2: Grid Preparation
    fld_path = None 
    reference_ligand_path = None

    if stage in ["all", "grid"]:
        if not quiet: print(f"\n[STAGE 2] Receptor & Grid Preparation")
        preparer = ReceptorPreparer()
        try:
            fld_path = preparer.prepare_receptor_and_grid(
                pdb_id, output_dir=grid_base_dir, allow_bad_res=True,
                ligand_resname=ligand_resname, protein_pdb_path=protein_pdb_path,
                ligand_pdb_path=ligand_pdb_path
            )
            
            # Reference ligand extraction/correction
            if ligand_pdb_path:
                 reference_ligand_path = grid_base_dir / "grid" / Path(ligand_pdb_path).name
            else:
                 reference_ligand_path = grid_base_dir / "grid" / f"{pdb_id}_ligand.pdb"
                 if ligand_resname:
                    try:
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
                                    reference_ligand_path = corrected_path
                    except: pass
        except Exception as e:
            result.status = WorkflowStatus.FAILED_GRID_PREP.value
            result.error_message = str(e)
            append_to_summary()
            return result

    # STAGE 3: Docking
    if stage in ["all", "docking"]:
        if not quiet: print(f"\n[STAGE 3] Docking Execution")
        raw_csv_path = work_dir / f"{data_prefix}_cleaned_data.csv"
        df = pl.read_csv(raw_csv_path)
        
        # Recovery of grid paths
        if not fld_path:
            fld_path = next(iter((grid_base_dir / "grid").glob("*.maps.fld")), None)
        if not reference_ligand_path:
            ref_corr = grid_base_dir / "grid" / f"{pdb_id}_ligand_corrected.sdf"
            reference_ligand_path = ref_corr if ref_corr.exists() else grid_base_dir / "grid" / f"{pdb_id}_ligand.pdb"

        try:
            oracle = AutoDockGPUOracle(
                receptor_file=fld_path, save_dir=work_dir / "docking_output",
                reference_ligand_path=reference_ligand_path, fragment_smiles=fragment_smiles,
                rmsd_threshold=rmsd_threshold, generate_isomers=not no_isomers
            )

            all_smiles = df.get_column("canonical_smiles").to_list()
            unique_smiles = list(set(all_smiles))
            smile_to_score = oracle.score_batch(unique_smiles)

            results_df = oracle.results_df
            # Merge and save
            results_df_renamed = results_df.rename({"smiles": "canonical_smiles"})
            df = df.join(results_df_renamed.select(["canonical_smiles", "docking_score", "score_valid", "score_best_any", "valid_pose_found", "n_conformers", "skip_reason"]),
                         on="canonical_smiles", how="left")
            df = df.with_columns(pl.col("valid_pose_found").fill_null(False))
            df.write_csv(work_dir / f"{data_prefix}_results_full.csv")

            # Update result metadata
            result.n_compounds_matched_2d = len(results_df.filter(pl.col("fragment_precheck") == True))
            result.n_compounds_docked = len(results_df.filter(pl.col("skip_reason").is_null()))
            result.n_valid_poses = len(results_df.filter(pl.col("valid_pose_found") == True))
            result.n_conformers_docked = int(results_df["n_conformers"].sum())

        except Exception as e:
            result.status = WorkflowStatus.FAILED_DOCKING.value
            result.error_message = str(e)
            append_to_summary()
            return result

    # STAGE 4: Analysis
    if stage in ["all", "analysis", "docking"]:
        print(f"\n[STAGE 4] Analysis & Plotting")
        detailed_path = work_dir / f"{data_prefix}_results_full.csv"
        df = pl.read_csv(detailed_path)
        
        plot_act_col = "standard_value" # Default plotted column
        plot_units = None if "pchembl_value" in df.columns else activity_units
        
        plot_docking_results(df, score_col="score_best_any", activity_col=plot_act_col, 
                             output_path=str(work_dir / f"{data_prefix}_analysis_best_any.png"), activity_units=plot_units)
        plot_docking_results(df, score_col="docking_score", activity_col=plot_act_col, 
                             output_path=str(work_dir / f"{data_prefix}_analysis_rmsd_constrained.png"), activity_units=plot_units)

    append_to_summary()
    return result

def main():
    parser = ArgumentParser(description="Run Docking Workflow from YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "retrieve", "grid", "docking", "analysis"])
    parser.add_argument("--smarts", type=str, help="Override SMARTS string for fragment filtering.")
    parser.add_argument("--no_isomers", action="store_true", help="Disable stereoisomer generation.")
    parser.add_argument("--run-dir", type=str, help="Base directory for the benchmark run.")
    parser.add_argument("--quiet", action="store_true", help="Minimal output.")

    args = parser.parse_args()
    config = load_config(args.config)
    
    result = run_docking_workflow(
        config=config,
        stage=args.stage,
        smarts=args.smarts,
        no_isomers=args.no_isomers,
        run_dir=args.run_dir,
        config_file_path=args.config,
        quiet=args.quiet
    )
    
    if result.status != WorkflowStatus.SUCCESS.value:
        if not args.quiet:
            print(f"\nWorkflow FAILED with status: {result.status}")
            if result.error_message:
                print(f"Error: {result.error_message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
