# Standard library imports
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing
import tempfile
import math

# Third-party imports
import numpy as np
import polars as pl
import yaml
from rdkit import Chem

# Local imports
from .data import fetch_chembl_data
from .docking import AutoDockGPUOracle
from .receptor import ReceptorPreparer
from .ligand_prep import LigandPreparer
from .analysis import DockingAnalyzer

class FCGMBOracle:
    """
    Fragment-Constrained Generative Model Benchmark (FCGMB) Oracle.
    
    Provides a standardized interface for benchmarking generative models
    against specific protein-ligand systems using fragment-constrained docking.
    """
    
    def __init__(
        self, 
        benchmark_name: str, 
        budget: int = 5000, 
        adgpu_executable: str = "adgpu",
        scratch_dir: Optional[Union[str, Path]] = None,
        n_cpus: Optional[int] = None,
        n_gpus: Optional[int] = 1
    ):
        """
        Initialize the oracle for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'CHEMBL205_1YDA_CHEMBL2331308')
            budget: Total number of compounds allowed to be scored.
            adgpu_executable: Path to the AutoDock-GPU executable.
            scratch_dir: Directory to store benchmark data (grids, results, etc.). 
            n_cpus: Number of CPUs for parallel operations.
            n_gpus: Number of GPUs for docking.
        """
        self.benchmark_name = benchmark_name
        self.max_budget = budget
        self.budget_used = 0
        self.finished = False
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.n_gpus = n_gpus or 1
        
        # Load configuration from internal package directory
        internal_config_dir = Path(__file__).parent / "configs"
        config_path = internal_config_dir / f"{benchmark_name}.yaml"
        
        if not config_path.exists():
            config_path = internal_config_dir / benchmark_name
            if not config_path.exists():
                config_path = Path("configs") / f"{benchmark_name}.yaml"
                if not config_path.exists():
                    config_path = Path("configs") / benchmark_name
                    
                if not config_path.exists():
                    available = self.list_benchmarks()
                    raise FileNotFoundError(f"Benchmark config '{benchmark_name}' not found.")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.target_id = self.config.get("target_id")
        self.pdb_id = self.config.get("pdb_id")
        self.doc_id = self.config.get("doc_id")
        self.fragment_smiles = self.config.get("fragment_smiles")
        self.rmsd_threshold = self.config.get("rmsd_threshold", 2.0)
        self.ligand_resname = self.config.get("ligand_resname")
        self.activity_units = self.config.get("activity_units", "nM")
        
        # Normalization bounds
        self.low_score = self.config.get("low_score")
        self.high_score = self.config.get("high_score")
        
        # Data storage organization
        if scratch_dir:
            self.scratch_dir = Path(scratch_dir).resolve()
        else:
            self.scratch_dir = Path.cwd() / ".fcgmb"
            
        self.grids_base_dir = self.scratch_dir / "grids"
        self.grid_dir = self.grids_base_dir / self.pdb_id
        self.ligand_data_dir = self.scratch_dir / "data"
        self.benchmark_run_dir = self.scratch_dir / "runs" / benchmark_name
        self.results_dir = self.benchmark_run_dir / "results"
        
        self.grid_dir.mkdir(parents=True, exist_ok=True)
        self.ligand_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FCGMB] Initialized benchmark: {benchmark_name}")
        
        self.docking_oracle = None
        self.ligand_preparer = None
        self.docking_analyzer = None
        self.adgpu_executable = adgpu_executable
        self.results_df = pl.DataFrame()

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        config_dir = Path(__file__).parent / "configs"
        if not config_dir.exists():
            return []
        return sorted([f.stem for f in config_dir.glob("*.yaml")])

    def get_initial_compounds(self) -> pl.DataFrame:
        """
        Retrieves the initial set of compounds for the benchmark.
        These are compounds from the document with bioactivity values in the lowest quartile.
        Caches the data in ligand_data directory.
        """
        df, threshold, act_col = self._get_full_data_and_threshold()
        if df.is_empty():
            return df
            
        initial_df = df.filter(pl.col(act_col) <= threshold)
        print(f"[FCGMB] Prepared {len(initial_df)} initial compounds (threshold {act_col} <= {threshold:.2f})")
        return initial_df

    def get_validation_compounds(self) -> pl.DataFrame:
        """
        Retrieves the validation set of compounds for the benchmark.
        These are compounds from the document with bioactivity values ABOVE the lowest quartile.
        """
        df, threshold, act_col = self._get_full_data_and_threshold()
        if df.is_empty():
            return df
            
        validation_df = df.filter(pl.col(act_col) > threshold)
        print(f"[FCGMB] Prepared {len(validation_df)} validation compounds (threshold {act_col} > {threshold:.2f})")
        return validation_df

    def _get_full_data_and_threshold(self) -> Tuple[pl.DataFrame, float, str]:
        """Internal helper to load data and compute the 25% threshold."""
        cache_file = self.ligand_data_dir / f"{self.benchmark_name}_chembl.csv"
        
        if cache_file.exists():
            print(f"[FCGMB] Loading cached ChEMBL data from {cache_file.name}")
            df = pl.read_csv(cache_file)
        else:
            print(f"[FCGMB] Downloading bioactivity data from ChEMBL for target {self.target_id}...")
            df = fetch_chembl_data(self.target_id, self.doc_id, units=self.activity_units)
            if not df.is_empty():
                df.write_csv(cache_file)
                print(f"[FCGMB] Saved ChEMBL data to {cache_file}")
        
        if df.is_empty():
            print("[FCGMB] Warning: No compounds found for this benchmark.")
            return df, 0.0, ""
            
        # Preferred activity column
        act_col = "pchembl_value" if "pchembl_value" in df.columns else "standard_value"
        
        # Filter: pchembl value below 25% threshold (based on range)
        pvals = df.get_column(act_col).to_numpy()
        min_v, max_v = np.min(pvals), np.max(pvals)
        threshold = min_v + 0.25 * (max_v - min_v)
        
        return df, threshold, act_col

    def _ensure_components(self):
        """Initialize all components and prepare grid if necessary."""
        if self.docking_oracle is not None:
            return

        # 1. Prepare Receptor/Grid
        fld_files = list(self.grid_dir.glob("*.maps.fld"))
        if not fld_files:
            print(f"[FCGMB] Grid not found. Preparing receptor and protein-ligand grids for {self.pdb_id}...")
            preparer = ReceptorPreparer(
                autogrid_executable="autogrid4", 
                mk_prepare_receptor_executable="mk_prepare_receptor.py", 
                reduce2_executable="mmtbx.reduce2"
            )
            fld_path = preparer.prepare_receptor_and_grid(
                self.pdb_id,
                ligand_resname=self.ligand_resname,
                output_dir=self.grids_base_dir / self.pdb_id, 
                allow_bad_res=True
            )
        else:
            fld_path = fld_files[0]
            print(f"[FCGMB] Using existing grids for {self.pdb_id}")
            
        # 2. Setup Analyzer
        ref_path = self.grid_dir / f"{self.pdb_id}_ligand_corrected.sdf"
        if not ref_path.exists():
             ref_path = self.grid_dir / f"{self.pdb_id}_ligand.pdb"
        
        self.docking_analyzer = DockingAnalyzer(
            reference_ligand_path=ref_path if ref_path.exists() else None,
            fragment_smiles=self.fragment_smiles,
            rmsd_threshold=self.rmsd_threshold
        )

        # 3. Setup Preparer
        self.ligand_preparer = LigandPreparer(n_cpus=self.n_cpus)

        # 4. Setup Docking Oracle
        print(f"[FCGMB] Initializing AutoDock-GPU Oracle...")
        self.docking_oracle = AutoDockGPUOracle(
            receptor_file=fld_path,
            adgpu_executable=self.adgpu_executable,
            save_dir=self.results_dir,
            n_cpus=self.n_cpus,
            n_gpus=self.n_gpus
        )

    def score(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Dock a list of SMILES and return their normalized scores [0.0 - 1.0].
        """
        if self.finished:
            print("[FCGMB] Oracle budget exhausted.")
            return {smi: 0.0 for smi in smiles_list}

        self._ensure_components()
        
        # 1. Pre-filtering (2D match) and initialization
        valid_compounds = []
        final_scores = {smi: 0.0 for smi in smiles_list}
        skipped_results = []
        
        for smi in smiles_list:
            if not self.docking_analyzer.check_2d_fragment_match(smi):
                skipped_results.append({
                    "smiles": smi,
                    "docking_score": float('nan'),
                    "normalized_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "best_any_score": float('nan'),
                    "skip_reason": "2D fragment mismatch",
                    "n_conformers": 0
                })
            else:
                valid_compounds.append(smi)
            
        if not valid_compounds:
            self._update_results_df(skipped_results)
            return final_scores

        # 2. Budget check
        remaining = self.max_budget - self.budget_used
        if remaining <= 0:
            self.finished = True
            return final_scores
            
        process_list = valid_compounds[:remaining]
        
        # 3. Prepare Ligands
        with tempfile.TemporaryDirectory(prefix="fcgmb_prep_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            smiles_to_pdbqts = self.ligand_preparer.prepare_batch(process_list, tmp_path)
            
            # 4. Dock
            docking_raw_results = self.docking_oracle.dock_batch(smiles_to_pdbqts, chunk_idx=self.budget_used)
            
            # 5. Analyze and Store Results
            smi_to_state_results = {}
            for res in docking_raw_results:
                smi = res["smiles"]
                if smi not in smi_to_state_results:
                    smi_to_state_results[smi] = []
                smi_to_state_results[smi].append(res)
            
            batch_results = []
            for smi in process_list:
                states = smi_to_state_results.get(smi, [])
                
                best_smi_valid_score = float('nan')
                best_smi_norm_score = 0.0
                best_smi_any_score = float('nan')
                best_smi_dlg = None
                valid_pose_found = False
                n_states = len(states)
                
                for state in states:
                    dlg_path = state["dlg_path"]
                    if not dlg_path: continue
                    
                    best_v, passed, best_m, best_a, best_am = self.docking_analyzer.filter_poses_by_rmsd(dlg_path, smi)
                    
                    if passed:
                        valid_pose_found = True
                        if math.isnan(best_smi_valid_score) or best_v < best_smi_valid_score:
                            best_smi_valid_score = best_v
                            best_smi_dlg = str(dlg_path)
                    
                    if math.isnan(best_smi_any_score) or best_a < best_smi_any_score:
                        best_smi_any_score = best_a
                
                # Calculate normalization for the best valid score of this SMILES
                if valid_pose_found:
                    if self.low_score is not None and self.high_score is not None:
                        denom = self.low_score - self.high_score
                        if abs(denom) > 1e-6:
                            best_smi_norm_score = (self.low_score - best_smi_valid_score) / denom
                        else:
                            best_smi_norm_score = 1.0 if best_smi_valid_score <= self.high_score else 0.0
                    else:
                        best_smi_norm_score = 0.0
                
                final_scores[smi] = best_smi_norm_score
                batch_results.append({
                    "smiles": smi,
                    "docking_score": best_smi_valid_score,
                    "normalized_score": best_smi_norm_score,
                    "valid_pose_found": valid_pose_found,
                    "dlg_path": best_smi_dlg,
                    "best_any_score": best_smi_any_score,
                    "skip_reason": None,
                    "n_conformers": n_states
                })

            # Update budget and combine results
            self.budget_used += len(process_list)
            all_batch_results = skipped_results + batch_results
            self._update_results_df(all_batch_results)

        if self.budget_used >= self.max_budget:
            self.finished = True
            
        return final_scores

    def _update_results_df(self, new_results: List[Dict]):
        """Helper to append new results to the main results dataframe."""
        new_df = pl.DataFrame(new_results)
        if self.results_df.is_empty():
            self.results_df = new_df
        else:
            # Ensure same columns before concat
            self.results_df = pl.concat([self.results_df, new_df])

    @property
    def status(self) -> str:
        return "finished" if self.finished else "active"

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_budget - self.budget_used)

    @property
    def fragment(self) -> str:
        """Returns the fragment SMILES that molecules must adhere to."""
        return self.fragment_smiles
