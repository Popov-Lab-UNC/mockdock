# Standard library imports
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
import tempfile
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
    from .ligand_prep import LigandPreparer
    from .analysis import DockingAnalyzer
    from .utils import (
        assign_bond_orders_from_template,
        fetch_ligand_expo_sdf,
        plot_activity_distribution,
        plot_docking_results,
    )
except ImportError:
    from fcgmb.data import fetch_chembl_data
    from fcgmb.docking import AutoDockGPUOracle
    from fcgmb.receptor import GridPrepError, LigandNotFoundError, PDBDownloadError, ReceptorPreparer
    from fcgmb.ligand_prep import LigandPreparer
    from fcgmb.analysis import DockingAnalyzer
    from fcgmb.utils import (
        assign_bond_orders_from_template,
        fetch_ligand_expo_sdf,
        plot_activity_distribution,
        plot_docking_results,
    )

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')
# Silence general warnings
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
    quiet: bool = False,
    n_cpus: Optional[int] = None,
    n_gpus: int = 1
) -> WorkflowResult:
    """
    Core function to run the docking workflow using refactored components.
    """
    start_time = time.time()

    # 1. Configuration extraction
    target_id = config.get("target_id")
    doc_id = config.get("doc_id")
    pdb_id = config.get("pdb_id")
    ligand_resname = config.get("ligand_resname")
    activity_units = config.get("activity_units", "nM")
    activity_col = config.get("activity_column", "standard_value")

    if run_dir:
        run_base = Path(run_dir)
        run_base.mkdir(exist_ok=True, parents=True)
        target_pdb_name = f"{target_id}_{pdb_id}"
        target_pdb_dir = run_base / target_pdb_name
        target_pdb_dir.mkdir(exist_ok=True, parents=True)
        work_dir = target_pdb_dir / str(doc_id) if doc_id else target_pdb_dir
        grid_base_dir = target_pdb_dir
    else:
        target_pdb_name = f"{target_id}_{pdb_id}"
        work_dir = Path(config.get("output_dir", f"{target_pdb_name}_workflow"))
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

    def append_to_summary():
        if not run_dir: return
        summary_path = Path(run_dir) / "benchmark_summary.csv"
        result.runtime_seconds = time.time() - start_time
        df_row = pl.DataFrame([asdict(result)])
        header = not summary_path.exists()
        with open(summary_path, "a") as f:
            df_row.write_csv(f, include_header=header)

    data_prefix = f"{target_id}_{pdb_id}_{doc_id}" if target_id and pdb_id and doc_id else "workflow_data"

    if not quiet:
        print(f"\n--- Starting Docking Workflow: {pdb_id} ---")

    # STAGE 1: Retrieval
    if stage in ["all", "retrieve"]:
        try:
            if config.get("ligand_csv_path"):
                df = pl.read_csv(Path(config.get("ligand_csv_path")))
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
                append_to_summary()
                return result
            
            df.write_csv(work_dir / f"{data_prefix}_cleaned_data.csv")
        except Exception as e:
            result.status = WorkflowStatus.FAILED_RETRIEVAL.value
            print(f"FAILED_RETRIEVAL: {e}", file=sys.stderr)
            append_to_summary()
            return result

    # STAGE 2: Grid Preparation
    fld_path = None 
    if stage in ["all", "grid"]:
        preparer = ReceptorPreparer()
        try:
            fld_path = preparer.prepare_receptor_and_grid(
                pdb_id, ligand_resname=ligand_resname, output_dir=grid_base_dir, 
                allow_bad_res=True, protein_pdb_path=protein_pdb_path,
                ligand_pdb_path=ligand_pdb_path
            )
        except Exception as e:
            result.status = WorkflowStatus.FAILED_GRID_PREP.value
            print(f"FAILED_GRID_PREP: {e}", file=sys.stderr)
            append_to_summary()
            return result

    # STAGE 3: Docking
    if stage in ["all", "docking"]:
        df = pl.read_csv(work_dir / f"{data_prefix}_cleaned_data.csv")
        if not fld_path:
            fld_path = next(iter(grid_base_dir.glob("*.maps.fld")), None)
        
        ref_corr = grid_base_dir / f"{pdb_id}_ligand_corrected.sdf"
        reference_ligand_path = ref_corr if ref_corr.exists() else grid_base_dir / f"{pdb_id}_ligand.pdb"

        try:
            # New modular setup
            preparer = LigandPreparer(n_cpus=n_cpus, generate_isomers=not no_isomers)
            oracle = AutoDockGPUOracle(receptor_file=fld_path, save_dir=work_dir / "results", n_cpus=n_cpus, n_gpus=n_gpus)
            analyzer = DockingAnalyzer(reference_ligand_path=reference_ligand_path, fragment_smiles=fragment_smiles, rmsd_threshold=rmsd_threshold)

            all_smiles = df.get_column("canonical_smiles").unique().to_list()
            
            # 2D filtering
            valid_smiles = [s for s in all_smiles if analyzer.check_2d_fragment_match(s)]
            result.n_compounds_matched_2d = len(valid_smiles)
            
            # Prepare and Dock
            with tempfile.TemporaryDirectory(prefix="workflow_prep_") as tmp_dir:
                smiles_to_pdbqts = preparer.prepare_batch(valid_smiles, Path(tmp_dir))
                docking_results = oracle.dock_batch(smiles_to_pdbqts, chunk_idx=0)
                
                # Analyze
                final_rows = []
                for res in docking_results:
                    smi = res["smiles"]
                    dlg_path = res["dlg_path"]
                    if dlg_path:
                        best_score, passed, best_mol, best_any_score, best_any_mol = analyzer.filter_poses_by_rmsd(dlg_path, smi)
                        final_rows.append({
                            "canonical_smiles": smi,
                            "docking_score": best_score if passed else float('nan'),
                            "score_valid": best_score if passed else float('nan'),
                            "score_best_any": best_any_score,
                            "valid_pose_found": passed,
                            "dlg_path": str(dlg_path)
                        })
                
                res_df = pl.DataFrame(final_rows)
                df = df.join(res_df, on="canonical_smiles", how="left")
                df = df.with_columns(pl.col("valid_pose_found").fill_null(False))
                df.write_csv(work_dir / f"{data_prefix}_results_full.csv")

                result.n_compounds_docked = len(res_df)
                result.n_valid_poses = len(res_df.filter(pl.col("valid_pose_found") == True))

        except Exception as e:
            result.status = WorkflowStatus.FAILED_DOCKING.value
            print(f"FAILED_DOCKING: {e}", file=sys.stderr)
            append_to_summary()
            return result

    # STAGE 4: Analysis
    if stage in ["all", "analysis", "docking"]:
        df = pl.read_csv(work_dir / f"{data_prefix}_results_full.csv")
        plot_units = None if "pchembl_value" in df.columns else activity_units
        plot_docking_results(df, score_col="docking_score", activity_col=activity_col, output_path=str(work_dir / f"{data_prefix}_results.png"), activity_units=plot_units)

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
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")

    args = parser.parse_args()
    config = load_config(args.config)
    
    result = run_docking_workflow(
        config=config, stage=args.stage, smarts=args.smarts, no_isomers=args.no_isomers,
        run_dir=args.run_dir, config_file_path=args.config, quiet=args.quiet,
        n_cpus=args.cpus, n_gpus=args.gpus
    )
    
    if result.status != WorkflowStatus.SUCCESS.value:
        sys.exit(1)

if __name__ == "__main__":
    main()
