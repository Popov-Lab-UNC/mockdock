#!/usr/bin/env python3
"""
Standalone script to run the docking workflow for specific protein-ligand benchmarks.
Relocated from fcgmb.workflow to allow cleaner package structure.
"""

# Standard library imports
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

# Third-party imports
import polars as pl
import yaml
from rdkit import Chem, RDLogger

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports (using absolute imports since this is a script)
from typing import cast

from fcgmb.analysis import DockingAnalyzer, aggregate_results_per_id
from fcgmb.data import fetch_chembl_data
from fcgmb.docking import AutoDockGPUOracle, AutoDockVinaOracle
from fcgmb.ligand_prep import LigandPreparer
from fcgmb.receptor import ReceptorPreparer
from fcgmb.utils import (
    check_2d_match,
    detect_gpus,
    plot_docking_results,
    resolve_backend,
)

# Silence RDKit noise
RDLogger.logger().setLevel(RDLogger.CRITICAL)
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
    assay_id: str
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
    with open(config_path) as f:
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
    n_gpus: Optional[int] = None,
    docking_backend: str = "auto",
) -> WorkflowResult:
    """
    Core function to run the docking workflow.
    """
    start_time = time.time()

    # 1. Configuration extraction
    target_id: str = config.get("target_id", "")
    doc_id: str = config.get("doc_id", "")
    assay_id: str = config.get("assay_id", "")
    pdb_id: str = config.get("pdb_id", "")
    ligand_resname: str = config.get("ligand_resname", "")
    activity_col = "pchembl_value"

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
        assay_id=assay_id or "",
        fragment_smiles=fragment_smiles or "",
        status=WorkflowStatus.SUCCESS.value,
    )

    def append_to_summary():
        if not run_dir:
            return
        summary_path = Path(run_dir) / "benchmark_summary.csv"
        result.runtime_seconds = time.time() - start_time
        df_row = pl.DataFrame([asdict(result)])
        header = not summary_path.exists()
        with open(summary_path, "a") as f:
            df_row.write_csv(f, include_header=header)

    data_prefix = f"{target_id}_{pdb_id}_{doc_id}"
    if assay_id:
        data_prefix += f"_{assay_id}"

    if not quiet:
        print(f"\n--- Starting Docking Workflow: {pdb_id} ---")

    # STAGE 1: Retrieval
    if stage in ["all", "retrieve"]:
        try:
            ligand_csv_path = config.get("ligand_csv_path")
            if ligand_csv_path:
                df = pl.read_csv(Path(cast(str, ligand_csv_path)))
                if "canonical_smiles" not in df.columns:
                    for col in ["smiles", "SMILES", "SMILE"]:
                        if col in df.columns:
                            df = df.rename({col: "canonical_smiles"})
                            break
                result.n_compounds_total = len(df)
            else:
                cache_dir = cast(Optional[str], config.get("chembl_cache_dir")) or os.environ.get("CHEMBL_CACHE_DIR")
                cache_only = bool(config.get("chembl_cache_only", False))
                df, stats = fetch_chembl_data(
                    target_id,
                    doc_id,
                    assay_chembl_id=assay_id,
                    return_stats=True,
                    cache_dir=cache_dir,
                    use_cache=True,
                    cache_only=cache_only,
                )
                # Use deduplicated count for total, as that's our real search space
                stats_dict = cast(dict, stats)
                result.n_compounds_total = stats_dict.get("n_deduplicated", 0)
                result.n_compounds_standardized = stats_dict.get("n_standardized", 0)

            if df is None or df.is_empty():
                result.status = WorkflowStatus.FAILED_RETRIEVAL.value
                append_to_summary()
                return result

            if isinstance(df, pl.DataFrame):
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
                pdb_id,
                ligand_resname=ligand_resname,
                output_dir=grid_base_dir,
                allow_bad_res=True,
                protein_pdb_path=protein_pdb_path,
                ligand_pdb_path=ligand_pdb_path,
            )
        except Exception as e:
            result.status = WorkflowStatus.FAILED_GRID_PREP.value
            print(f"FAILED_GRID_PREP: {e}", file=sys.stderr)
            append_to_summary()
            return result

    # STAGE 3: Docking
    if stage in ["all", "docking"]:
        df_path = work_dir / f"{data_prefix}_cleaned_data.csv"
        if not df_path.exists():
            print(f"   Error: Cleaned data not found at {df_path}. Skipping docking.")
            result.status = WorkflowStatus.FAILED_DOCKING.value
            append_to_summary()
            return result

        df = pl.read_csv(df_path)
        if "pchembl_value" not in df.columns:
            raise RuntimeError("Missing pchembl_value in cleaned data; cannot proceed.")
        if df.is_empty():
            print("   Warning: No compounds to dock. Skipping.")
            result.status = WorkflowStatus.SUCCESS.value
            append_to_summary()
            return result

        if not fld_path:
            fld_path = next(iter(grid_base_dir.glob("*.maps.fld")), None)
        
        if fld_path is None:
             raise FileNotFoundError(f"Grid file (.maps.fld) not found in {grid_base_dir}")

        ref_corr = grid_base_dir / f"{pdb_id}_ligand_corrected.sdf"
        reference_ligand_path = (
            ref_corr if ref_corr.exists() else grid_base_dir / f"{pdb_id}_ligand.pdb"
        )

        try:
            # 1. Hardware and Backend Resolution
            actual_n_cpus = n_cpus or multiprocessing.cpu_count()
            actual_n_gpus = n_gpus if n_gpus is not None else detect_gpus()

            adgpu_exe = config.get("adgpu_executable", "adgpu")
            resolved = resolve_backend(
                requested_backend=docking_backend,
                n_gpus=actual_n_gpus,
                adgpu_executable=adgpu_exe,
            )

            print(
                f"   Docking Backend: {resolved.upper()} ({actual_n_cpus} CPUs, {actual_n_gpus} GPUs)"
            )

            # 2. Setup Components
            preparer = LigandPreparer(n_cpus=actual_n_cpus, generate_isomers=not no_isomers)

            if resolved == "autodock_gpu":
                oracle = AutoDockGPUOracle(
                    receptor_file=fld_path,
                    adgpu_executable=adgpu_exe,
                    save_dir=work_dir / "results",
                    n_cpus=actual_n_cpus,
                    n_gpus=actual_n_gpus,
                )
            else:
                extra_vina = config.get("vina_config", {})
                oracle = AutoDockVinaOracle(
                    receptor_file=fld_path,
                    save_dir=work_dir / "results",
                    exhaustiveness=extra_vina.get("exhaustiveness", 32),
                    n_poses=extra_vina.get("n_poses", 10),
                    n_cpus=actual_n_cpus,
                )

            analyzer = DockingAnalyzer(
                reference_ligand_path=reference_ligand_path,
                fragment_smiles=fragment_smiles,
                rmsd_threshold=rmsd_threshold,
            )

            all_smiles = df.get_column("canonical_smiles").unique().to_list()
            print(f"   Total unique compounds: {len(all_smiles)}")

            # 3. 2D filtering
            valid_smiles = [
                s
                for s in all_smiles
                if (m := Chem.MolFromSmiles(s)) is not None and check_2d_match(m, analyzer.fragment_mol)
            ]
            print(f"   Compounds matching 2D fragment '{fragment_smiles}': {len(valid_smiles)}")
            result.n_compounds_matched_2d = len(valid_smiles)

            if not valid_smiles:
                print("   Warning: No compounds matched 2D fragment. Skipping docking.")
                result.status = WorkflowStatus.SUCCESS.value
                append_to_summary()
                return result

            # 4. Prepare and Dock
            print(f"   Preparing and docking {len(valid_smiles)} compounds...")
            with tempfile.TemporaryDirectory(prefix="workflow_prep_") as tmp_dir:
                smiles_to_pdbqts = preparer.prepare_batch(valid_smiles, Path(tmp_dir))
                docking_results = oracle.dock_batch(smiles_to_pdbqts, chunk_idx=0)

                # Analyze
                final_rows = []
                for res in docking_results:
                    smi = res["smiles"]
                    pose_path = res["dlg_path"]
                    if pose_path:
                        # Returns: (best_v, passed, best_v_mol, best_a, best_a_mol, best_v_idx, best_a_idx)
                        best_score, passed, best_mol, best_any_score, best_any_mol, _, _ = (
                            analyzer.filter_poses_by_rmsd(pose_path, smi)
                        )
                        final_rows.append(
                            {
                                "canonical_smiles": smi,
                                "docking_score": best_score if passed else float("nan"),
                                "score_valid": best_score if passed else float("nan"),
                                "score_best_any": best_any_score,
                                "valid_pose_found": passed,
                                "dlg_path": str(pose_path),
                            }
                        )

                res_df = pl.DataFrame(final_rows)
                if not res_df.is_empty():
                    df = df.join(res_df, on="canonical_smiles", how="left")
                    df = df.with_columns(pl.col("valid_pose_found").fill_null(False))
                    df = aggregate_results_per_id(
                        df,
                        score_col="docking_score",
                        valid_col="valid_pose_found",
                        activity_col=activity_col,
                    )
                    df.write_csv(work_dir / f"{data_prefix}_results.csv")

                    result.n_compounds_docked = len(res_df)
                    result.n_valid_poses = len(res_df.filter(pl.col("valid_pose_found")))
                    print(
                        f"   Docking complete. Docked {len(res_df)} conformers, {result.n_valid_poses} passed RMSD."
                    )
                else:
                    print("   Warning: No docking results obtained from oracle.")
                    result.status = WorkflowStatus.FAILED_DOCKING.value
                    append_to_summary()
                    return result

        except Exception as e:
            result.status = WorkflowStatus.FAILED_DOCKING.value
            print(f"FAILED_DOCKING: {e}", file=sys.stderr)
            traceback.print_exc()
            append_to_summary()
            return result

    # STAGE 4: Analysis
    if stage in ["all", "analysis", "docking"]:
        results_file = work_dir / f"{data_prefix}_results.csv"
        if results_file.exists():
            df = pl.read_csv(results_file)
            if "pchembl_value" not in df.columns:
                raise RuntimeError("Missing pchembl_value in results; cannot plot.")
            plot_docking_results(
                df,
                score_col="docking_score",
                activity_col=activity_col,
                output_path=str(work_dir / f"{data_prefix}_results.png"),
            )

    append_to_summary()
    return result


def main():
    parser = ArgumentParser(description="Run Docking Workflow from YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "retrieve", "grid", "docking", "analysis"],
    )
    parser.add_argument("--smarts", type=str, help="Override SMARTS string for fragment filtering.")
    parser.add_argument(
        "--no_isomers", action="store_true", help="Disable stereoisomer generation."
    )
    parser.add_argument("--run-dir", type=str, help="Base directory for the benchmark run.")
    parser.add_argument("--quiet", action="store_true", help="Minimal output.")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use.")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "autodock_gpu", "vina"],
        help="Docking software backend.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/chembl_cache",
        help="ChEMBL data cache directory",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    if args.cache_dir:
        config["chembl_cache_dir"] = args.cache_dir

    result = run_docking_workflow(
        config=config,
        stage=args.stage,
        smarts=args.smarts,
        no_isomers=args.no_isomers,
        run_dir=args.run_dir,
        config_file_path=args.config,
        quiet=args.quiet,
        n_cpus=args.cpus,
        n_gpus=args.gpus,
        docking_backend=args.backend,
    )

    if result.status != WorkflowStatus.SUCCESS.value:
        sys.exit(1)


if __name__ == "__main__":
    main()
