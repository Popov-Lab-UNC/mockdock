import os
import yaml
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from rdkit import Chem
from .docking import AutoDockGPUOracle
from .data import fetch_chembl_data
from .receptor import ReceptorPreparer

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
        scratch_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the oracle for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'CHEMBL205_1YDA_CHEMBL2331308')
            budget: Total number of compounds allowed to be scored.
            adgpu_executable: Path to the AutoDock-GPU executable.
            scratch_dir: Directory to store benchmark data (grids, results, etc.). 
                         Defaults to '.fcgmb' in the current working directory.
        """
        self.benchmark_name = benchmark_name
        self.max_budget = budget
        self.budget_used = 0
        self.finished = False
        
        # Load configuration from internal package directory
        internal_config_dir = Path(__file__).parent / "configs"
        config_path = internal_config_dir / f"{benchmark_name}.yaml"
        
        if not config_path.exists():
            # Try without .yaml extension
            config_path = internal_config_dir / benchmark_name
            if not config_path.exists():
                # Fallback to current directory for user-provided configs
                config_path = Path("configs") / f"{benchmark_name}.yaml"
                if not config_path.exists():
                    config_path = Path("configs") / benchmark_name
                    
                if not config_path.exists():
                    available = self.list_benchmarks()
                    raise FileNotFoundError(
                        f"Benchmark config '{benchmark_name}' not found.\n"
                        f"Looked in: {internal_config_dir}\n"
                        f"Available internal benchmarks: {available}"
                    )
        
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
        
        # Prepare fragment molecule for pre-checks
        self.fragment_mol = None
        if self.fragment_smiles:
            self.fragment_mol = Chem.MolFromSmiles(self.fragment_smiles)
            if self.fragment_mol is None:
                # Fallback to SMARTS if SMILES fails (e.g. for general patterns)
                self.fragment_mol = Chem.MolFromSmarts(self.fragment_smiles)
        
        # Data storage organization
        if scratch_dir:
            self.scratch_dir = Path(scratch_dir).resolve()
        else:
            self.scratch_dir = Path.cwd() / ".fcgmb"
            
        # Specific subdirectories as requested
        self.grids_base_dir = self.scratch_dir / "grids"
        # ReceptorPreparer adds "grid" to the output_dir, so we define grid_dir accordingly
        self.grid_dir = self.grids_base_dir / self.pdb_id / "grid"
        self.ligand_data_dir = self.scratch_dir / "ligand_data"
        self.benchmark_run_dir = self.scratch_dir / "benchmarks" / benchmark_name
        self.results_dir = self.benchmark_run_dir / "results"
        
        # Ensure directories exist
        self.grid_dir.mkdir(parents=True, exist_ok=True)
        self.ligand_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FCGMB] Initialized benchmark: {benchmark_name}")
        print(f"[FCGMB] Scratch directory: {self.scratch_dir}")
        
        self.oracle = None
        self.adgpu_executable = adgpu_executable

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all available benchmarks in the internal configs directory."""
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
            return df
            
        # Preferred activity column
        act_col = "pchembl_value" if "pchembl_value" in df.columns else "standard_value"
        
        # Filter: pchembl value below 25% threshold (based on range)
        pvals = df.get_column(act_col).to_numpy()
        min_v, max_v = np.min(pvals), np.max(pvals)
        threshold = min_v + 0.25 * (max_v - min_v)
        
        initial_df = df.filter(pl.col(act_col) <= threshold)
        print(f"[FCGMB] Prepared {len(initial_df)} initial compounds (threshold {act_col} <= {threshold:.2f})")
        return initial_df

    def _ensure_oracle(self):
        """Initialize the AutoDock-GPU Oracle and prepare grid if necessary."""
        if self.oracle is not None:
            return

        # Check if grid exists
        fld_files = list(self.grid_dir.glob("*.maps.fld"))
        if not fld_files:
            print(f"[FCGMB] Grid not found. Preparing receptor and protein-ligand grids for {self.pdb_id}...")
            preparer = ReceptorPreparer()
            fld_path = preparer.prepare_receptor_and_grid(
                self.pdb_id,
                output_dir=self.grids_base_dir / self.pdb_id, 
                allow_bad_res=True,
                ligand_resname=self.ligand_resname
            )
        else:
            fld_path = fld_files[0]
            print(f"[FCGMB] Using existing grids for {self.pdb_id}")
            
        # Reference ligand for RMSD
        ref_path = self.grid_dir / f"{self.pdb_id}_ligand_corrected.sdf"
        if not ref_path.exists():
             ref_path = self.grid_dir / f"{self.pdb_id}_ligand.pdb"
             
        # Check sub-grid dir just in case (legacy or standard behavior)
        if not ref_path.exists():
            ref_path = self.grid_dir / "grid" / f"{self.pdb_id}_ligand_corrected.sdf"
            if not ref_path.exists():
                ref_path = self.grid_dir / "grid" / f"{self.pdb_id}_ligand.pdb"

        print(f"[FCGMB] Initializing AutoDock-GPU Oracle...")
        self.oracle = AutoDockGPUOracle(
            receptor_file=fld_path,
            adgpu_executable=self.adgpu_executable,
            save_dir=self.results_dir,
            reference_ligand_path=ref_path if ref_path.exists() else None,
            fragment_smiles=self.fragment_smiles,
            rmsd_threshold=self.rmsd_threshold
        )

    def score(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Dock a list of SMILES and return their scores.
        Only docks if the budget has not been exceeded and the molecule matches the constraint.
        """
        if self.finished:
            print("[FCGMB] Oracle budget exhausted. Returning 0.0 for all scores.")
            return {smi: 0.0 for smi in smiles_list}

        self._ensure_oracle()
        
        # 1. Pre-evaluation: Substructure match and SMILES validity
        print(f"[FCGMB] Evaluating {len(smiles_list)} compounds for substructure match...")
        
        valid_compounds = []
        invalid_results = {}
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_results[smi] = 0.0
                continue
                
            if self.fragment_mol and not mol.HasSubstructMatch(self.fragment_mol):
                invalid_results[smi] = 0.0
                continue
            
            valid_compounds.append(smi)
            
        n_invalid = len(smiles_list) - len(valid_compounds)
        if n_invalid > 0:
            print(f"[FCGMB] Warning: {n_invalid} compounds are invalid or do not match the fragment constraint. Returning 0.0 for these (not counted against budget).")

        if not valid_compounds:
            return invalid_results

        # 2. Budget check and docking for valid compounds
        remaining = self.max_budget - self.budget_used
        if remaining <= 0:
            self.finished = True
            print("[FCGMB] Oracle budget exhausted. Returning 0.0 for remaining compounds.")
            for smi in valid_compounds:
                invalid_results[smi] = 0.0
            return invalid_results
            
        # Only process what fits in budget
        process_list = valid_compounds[:remaining]
        skipped_list = valid_compounds[remaining:]
        
        if skipped_list:
            print(f"[FCGMB] Warning: {len(skipped_list)} compounds exceeds budget and will not be docked.")

        print(f"[FCGMB] Scoring {len(process_list)} valid compounds (Budget used: {self.budget_used}/{self.max_budget})...")
        print(f"[FCGMB] Preparing Ligands for Docking and running AutoDock-GPU...")
        
        docking_results = self.oracle.score_batch(process_list)
        
        # Update budget with compounds that were actually sent to the docking engine
        self.budget_used += len(process_list)
        
        # Get detailed results to check for RMSD constraint
        detailed_results = self.oracle.results_df
        
        normalized_results = {}
        for row in detailed_results.to_dicts():
            smi = row["smiles"]
            raw_score = row["docking_score"]
            valid_pose = row.get("valid_pose_found", False)
            
            # Application of FCGMB scoring logic:
            # 1. 0.0 if RMSD constraint failed
            # 2. Normalized score if RMSD passed
            if not valid_pose or np.isnan(raw_score):
                final_score = 0.0
            else:
                # Normalization: (low - raw) / (low - high)
                # 1.0 = high_score (best), 0.0 = low_score (worst)
                if self.low_score is not None and self.high_score is not None:
                    denom = self.low_score - self.high_score
                    if abs(denom) > 1e-6:
                        final_score = (self.low_score - raw_score) / denom
                    else:
                        final_score = 1.0 if raw_score <= self.high_score else 0.0
                else:
                    # Fallback to raw if bounds not defined (though they should be now)
                    final_score = raw_score
            
            normalized_results[smi] = final_score
        
        # 3. Assemble final results
        # Merge invalid_results (already 0.0) with normalized_results
        final_results = invalid_results
        final_results.update(normalized_results)
        
        # Fill results with 0.0 for those beyond budget
        for smi in skipped_list:
            final_results[smi] = 0.0
            
        if self.budget_used >= self.max_budget:
            self.finished = True
            print("[FCGMB] Budget limit reached. Oracle status set to finished.")
            
        return final_results

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
