import os
import yaml
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from .docking import AutoDockGPUOracle
from .data import fetch_chembl_data
from .receptor import ReceptorPreparer

class FCGMBOracle:
    """
    Fragment-Constrained Generative Model Benchmark (FCGMB) Oracle.
    
    Provides a standardized interface for benchmarking generative models
    against specific protein-ligand systems using fragment-constrained docking.
    """
    
    def __init__(self, benchmark_name: str, budget: int = 5000, adgpu_executable: str = "adgpu"):
        """
        Initialize the oracle for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'CHEMBL205_1YDA_CHEMBL2331308')
            budget: Total number of compounds allowed to be scored.
            adgpu_executable: Path to the AutoDock-GPU executable.
        """
        self.benchmark_name = benchmark_name
        self.max_budget = budget
        self.budget_used = 0
        self.finished = False
        
        # Load configuration
        config_path = Path("configs") / f"{benchmark_name}.yaml"
        if not config_path.exists():
            # Try without .yaml extension
            config_path = Path("configs") / benchmark_name
            if not config_path.exists():
                raise FileNotFoundError(f"Benchmark config not found: {config_path}")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.target_id = self.config.get("target_id")
        self.pdb_id = self.config.get("pdb_id")
        self.doc_id = self.config.get("doc_id")
        self.fragment_smiles = self.config.get("fragment_smiles")
        self.rmsd_threshold = self.config.get("rmsd_threshold", 2.0)
        self.ligand_resname = self.config.get("ligand_resname")
        self.activity_units = self.config.get("activity_units", "nM")
        
        # Paths
        self.base_dir = Path("benchmarks") / benchmark_name
        self.grid_dir = self.base_dir / "grid"
        self.results_dir = self.base_dir / "results"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.oracle = None
        self.adgpu_executable = adgpu_executable

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all available benchmarks in the configs directory."""
        config_dir = Path("configs")
        if not config_dir.exists():
            return []
        return [f.stem for f in config_dir.glob("*.yaml")]

    def get_initial_compounds(self) -> pl.DataFrame:
        """
        Retrieves the initial set of compounds for the benchmark.
        These are compounds from the document with bioactivity values in the lowest quartile.
        """
        print(f"Fetching initial compounds for {self.benchmark_name}...")
        df = fetch_chembl_data(self.target_id, self.doc_id, units=self.activity_units)
        
        if df.is_empty():
            print("Warning: No compounds found for this benchmark.")
            return df
            
        # Preferred activity column
        act_col = "pchembl_value" if "pchembl_value" in df.columns else "standard_value"
        
        # Filter: pchembl value below 25% threshold (based on range)
        pvals = df.get_column(act_col).to_numpy()
        min_v, max_v = np.min(pvals), np.max(pvals)
        threshold = min_v + 0.25 * (max_v - min_v)
        
        initial_df = df.filter(pl.col(act_col) <= threshold)
        print(f"Found {len(initial_df)} initial compounds (threshold {act_col} <= {threshold:.2f})")
        return initial_df

    def _ensure_oracle(self):
        """Initialize the AutoDock-GPU Oracle and prepare grid if necessary."""
        if self.oracle is not None:
            return

        # Check if grid exists
        fld_files = list(self.grid_dir.glob("*.maps.fld"))
        if not fld_files:
            print(f"Grid not found in {self.grid_dir}. Preparing receptor...")
            preparer = ReceptorPreparer()
            fld_path = preparer.prepare_receptor_and_grid(
                self.pdb_id,
                output_dir=self.base_dir,
                allow_bad_res=True,
                ligand_resname=self.ligand_resname
            )
        else:
            fld_path = fld_files[0]
            
        # Reference ligand for RMSD
        # This mirrors run_workflow.py logic
        ref_path = self.grid_dir / f"{self.pdb_id}_ligand_corrected.sdf"
        if not ref_path.exists():
             ref_path = self.grid_dir / f"{self.pdb_id}_ligand.pdb"
             
        # Check sub-grid dir just in case (legacy)
        if not ref_path.exists():
            ref_path = self.grid_dir / "grid" / f"{self.pdb_id}_ligand_corrected.sdf"
            if not ref_path.exists():
                ref_path = self.grid_dir / "grid" / f"{self.pdb_id}_ligand.pdb"

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
        Only docks if the budget has not been exceeded.
        """
        if self.finished:
            return {smi: 0.0 for smi in smiles_list}

        self._ensure_oracle()
        
        remaining = self.max_budget - self.budget_used
        if remaining <= 0:
            self.finished = True
            return {smi: 0.0 for smi in smiles_list}
            
        # Only process what fits in budget
        process_list = smiles_list[:remaining]
        skipped_list = smiles_list[remaining:]
        
        print(f"Scoring {len(process_list)} compounds (Budget: {self.budget_used}/{self.max_budget})...")
        results = self.oracle.score_batch(process_list)
        
        self.budget_used += len(process_list)
        
        # Fill results with 0.0 for those beyond budget
        for smi in skipped_list:
            results[smi] = 0.0
            
        if self.budget_used >= self.max_budget:
            self.finished = True
            print("Budget limit reached. Oracle status set to finished.")
            
        return results

    @property
    def status(self) -> str:
        return "finished" if self.finished else "active"

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_budget - self.budget_used)
