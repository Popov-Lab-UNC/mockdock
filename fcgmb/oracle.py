# Standard library imports
import json
import math
import multiprocessing
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import polars as pl
import yaml
from rdkit import Chem

# Local imports
from .analysis import DockingAnalyzer
from .data import fetch_chembl_data
from .docking import AutoDockGPUOracle, AutoDockVinaOracle
from .ligand_prep import LigandPreparer
# ReceptorPreparer (requires prody) is imported lazily below — only needed
# when no pre-built grid is found.


def _detect_gpus() -> int:
    """Detect available NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return len([line for line in result.stdout.splitlines() if line.strip().startswith('GPU')])
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return 0


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
        docking_backend: str = "auto",
        scratch_dir: Optional[Union[str, Path]] = None,
        run_dir: Optional[Union[str, Path]] = None,
        n_cpus: Optional[int] = None,
        n_gpus: Optional[int] = None,
    ):
        """
        Initialize the oracle for a specific benchmark.

        Args:
            benchmark_name: Name of the benchmark (e.g. 'AKT1', 'CHK1', 'ITK',
                'PCK1', 'TTK', 'VEGFR2'). Run FCGMBOracle.list_benchmarks() for
                the full list.
            budget: Total number of compounds allowed to be scored.
            docking_backend: Backend to use ('autodock_gpu', 'vina', or 'auto').
            scratch_dir: Directory to store persistent cache assets (grids,
                bioactivity data). Defaults to ~/.fcgmb.
            run_dir: Directory for this run's outputs (docking results, live CSV,
                metrics, top poses). Defaults to ./run_<timestamp>/ in CWD.
            n_cpus: Number of CPUs for parallel operations. Autodetected if None.
            n_gpus: Number of GPUs for docking. Autodetected if None.
        """
        # ── Public state ──────────────────────────────────────────────
        self.benchmark_name = benchmark_name
        self.max_budget = budget
        self.budget_used = 0
        self._generation_round = 0
        self.pdb_id: Optional[str] = None   # set after config load below
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.n_gpus = n_gpus if n_gpus is not None else _detect_gpus()
        self.results_df = pl.DataFrame()
        # Timing accumulators
        self._total_prep_time = 0.0
        self._total_dock_time = 0.0
        self._total_analysis_time = 0.0
        self._n_prepped = 0
        self._n_docked = 0

        # ── Backend settings (private) ────────────────────────────────
        self._docking_backend = docking_backend
        self._resolved_backend: Optional[str] = None
        self._backend_config = {
            "adgpu_executable": "adgpu",
            "vina_exhaustiveness": 32,
            "n_poses": 10,
        }

        # ── Load config ───────────────────────────────────────────────
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
                    raise FileNotFoundError(
                        f"Benchmark config '{benchmark_name}' not found. "
                        f"Available: {available}"
                    )

        with open(config_path, "r") as f:
            _raw = yaml.safe_load(f)

        self.pdb_id = _raw.get("pdb_id")

        # Private config fields
        self._target_id = _raw.get("target_id")
        self._doc_id = _raw.get("doc_id")
        self._fragment_smiles = _raw.get("fragment_smiles")
        self._fragment_smiles_with_dummies = _raw.get("fragment_smiles_with_dummies")
        self._rmsd_threshold = _raw.get("rmsd_threshold", 2.0)
        self._require_fragment_match = _raw.get("require_fragment_match", True)
        self._require_pose_rmsd = _raw.get("require_pose_rmsd", True)
        # Invariant: without a 2D fragment match, fragment RMSD is undefined
        if not self._require_fragment_match:
            self._require_pose_rmsd = False
        self._ligand_resname = _raw.get("ligand_resname")
        self._low_score = _raw.get("low_score")
        self._high_score = _raw.get("high_score")

        # ── Directory layout (all private) ────────────────────────────
        _scratch = Path(scratch_dir).resolve() if scratch_dir else Path.home() / ".fcgmb"
        _pkg = Path(__file__).parent
        self._pkg_bioactivity_dir = _pkg / "bioactivity_data"
        self._pkg_grids_dir = _pkg / "grids"
        self._grids_base_dir = _scratch / "grids"
        self._grid_dir = self._grids_base_dir / self.pdb_id
        self._bioactivity_data_dir = _scratch / "bioactivity_data"
        # Lazy: cache dirs are created only when actually needed (before first ChEMBL
        # fetch or grid generation) so that a fresh install with pre-built grids and
        # bundled bioactivity data never creates directories unnecessarily.

        # ── Timestamped run directory ──────────────────────────────────
        # This is the visible output directory for all run-specific artifacts.
        # Unlike scratch_dir (which is a persistent cache), run_dir lives in the
        # user's current working directory so it is easy to find.
        if run_dir is not None:
            self._run_dir = Path(run_dir).resolve()
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = Path.cwd() / f"run_{timestamp}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir = self._run_dir / "docking_results"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FCGMB] Run directory: {self._run_dir}")

        # ── Lazy-initialised components (private) ─────────────────────
        self._docking_oracle = None
        self._ligand_preparer = None
        self._docking_analyzer = None
        self._chembl_data = None

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        """Timestamped run directory for this oracle session."""
        return self._run_dir

    @property
    def generation_round(self) -> int:
        """Number of score() batch calls made so far (i.e., generation rounds)."""
        return self._generation_round

    @property
    def fragment_smiles(self) -> str:
        """Fragment SMILES that every submitted molecule must contain."""
        return self._fragment_smiles

    @property
    def fragment_smiles_with_dummies(self) -> Optional[str]:
        """Fragment SMILES with (*) dummy attachment point(s) for PromptSMILES
        scaffold decoration.  Returns None if not yet set in the benchmark config YAML."""
        return self._fragment_smiles_with_dummies

    @property
    def config(self) -> dict:
        """Key benchmark configuration parameters."""
        return {
            "rmsd_threshold": self._rmsd_threshold,
            "require_fragment_match": self._require_fragment_match,
            "require_pose_rmsd": self._require_pose_rmsd,
            "low_score": self._low_score,
            "high_score": self._high_score,
        }

    @property
    def status(self) -> str:
        return "finished" if self.budget_used >= self.max_budget else "active"

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_budget - self.budget_used)

    def set_backend_config(self, **kwargs):
        """Override default backend settings (e.g. vina_exhaustiveness, n_poses)."""
        self._backend_config.update(kwargs)

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """Return all canonical benchmark names bundled with the fcgmb package."""
        pkg_config_dir = Path(__file__).parent / "configs"
        if not pkg_config_dir.exists():
            return []
        return sorted(f.stem for f in pkg_config_dir.glob("*.yaml"))

    def get_initial_compounds(self) -> pl.DataFrame:
        """
        Retrieve the initial compound set (lowest-quartile bioactivity).
        These are provided to the generative model as starting points.
        """
        df, threshold, act_col = self._get_full_data_and_threshold()
        if df.is_empty():
            return df
        initial_df = df.filter(pl.col(act_col) <= threshold)
        has_score = "score" in initial_df.columns
        print(
            f"[FCGMB] Prepared {len(initial_df)} initial compounds "
            f"(threshold {act_col} <= {threshold:.2f})"
            + (" [pre-computed docking scores available]" if has_score else "")
        )
        return initial_df

    def get_validation_compounds(self) -> pl.DataFrame:
        """
        Retrieve the validation compound set (above-lowest-quartile bioactivity).
        These are used to evaluate oracle performance.
        """
        df, threshold, act_col = self._get_full_data_and_threshold()
        if df.is_empty():
            return df
        validation_df = df.filter(pl.col(act_col) > threshold)
        print(
            f"[FCGMB] Prepared {len(validation_df)} validation compounds "
            f"(threshold {act_col} > {threshold:.2f})"
        )
        return validation_df

    def score(self, smiles_list: List[str]) -> Dict[str, float]:
        """Dock a list of SMILES and return normalised scores in [0.0, 1.0]."""
        self._generation_round += 1
        if self.budget_used >= self.max_budget:
            print("[FCGMB] Oracle budget exhausted.")
            return {smi: 0.0 for smi in smiles_list}

        self._ensure_components()

        # 1. Pre-filter: 2-D fragment match
        valid_compounds = []
        final_scores = {smi: 0.0 for smi in smiles_list}
        skipped_results = []

        for smi in smiles_list:
            if self._require_fragment_match and not self._docking_analyzer.check_2d_fragment_match(smi):
                skipped_results.append({
                    "smiles": smi,
                    "docking_score": float("nan"),
                    "normalized_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "best_any_score": float("nan"),
                    "skip_reason": "2D fragment mismatch",
                    "n_conformers": 0,
                })
            else:
                valid_compounds.append(smi)

        if not valid_compounds:
            self._update_results_df(skipped_results)
            return final_scores

        # 2. Budget cap
        process_list = valid_compounds[: self.max_budget - self.budget_used]

        # 3. Prepare → Dock → Analyse
        with tempfile.TemporaryDirectory(prefix="fcgmb_prep_") as tmp_dir:
            t0_prep = time.time()
            smiles_to_pdbqts = self._ligand_preparer.prepare_batch(
                process_list, Path(tmp_dir)
            )
            self._total_prep_time += time.time() - t0_prep
            self._n_prepped += len(process_list)

            t0_dock = time.time()
            docking_raw_results = self._docking_oracle.dock_batch(
                smiles_to_pdbqts, chunk_idx=self.budget_used
            )
            self._total_dock_time += time.time() - t0_dock
            self._n_docked += len(process_list)

            t0_analysis = time.time()

            smi_to_states: Dict[str, list] = {}
            for res in docking_raw_results:
                smi_to_states.setdefault(res["smiles"], []).append(res)

            batch_results = []
            for smi in process_list:
                states = smi_to_states.get(smi, [])
                best_valid = float("nan")
                best_norm = 0.0
                best_any = float("nan")
                best_dlg = None
                valid_pose_found = False

                for state in states:
                    dlg_path = state["dlg_path"]
                    if not dlg_path:
                        continue
                    best_v, passed, _bm, best_a, _bam = (
                        self._docking_analyzer.filter_poses_by_rmsd(dlg_path, smi)
                    )
                    if math.isnan(best_any) or best_a < best_any:
                        best_any = best_a
                    if self._require_pose_rmsd:
                        # Standard mode: only accept poses within RMSD threshold
                        if passed:
                            valid_pose_found = True
                            if math.isnan(best_valid) or best_v < best_valid:
                                best_valid = best_v
                                best_dlg = str(dlg_path)
                    else:
                        # Relaxed mode: accept any pose; use best overall score
                        if not math.isnan(best_a):
                            valid_pose_found = True
                            if math.isnan(best_valid) or best_a < best_valid:
                                best_valid = best_a
                                best_dlg = str(dlg_path)

                if valid_pose_found and self._low_score is not None and self._high_score is not None:
                    denom = self._low_score - self._high_score
                    if abs(denom) > 1e-6:
                        best_norm = (self._low_score - best_valid) / denom
                    else:
                        best_norm = 1.0 if best_valid <= self._high_score else 0.0

                final_scores[smi] = best_norm
                batch_results.append({
                    "smiles": smi,
                    "docking_score": best_valid,
                    "normalized_score": best_norm,
                    "valid_pose_found": valid_pose_found,
                    "dlg_path": best_dlg,
                    "best_any_score": best_any,
                    "skip_reason": None,
                    "n_conformers": len(states),
                })

            self._total_analysis_time += time.time() - t0_analysis

            self.budget_used += len(process_list)
            print(
                f"[FCGMB] Round {self._generation_round}: scored {len(process_list)} molecules "
                f"(budget {self.budget_used}/{self.max_budget})"
            )
            self._update_results_df(skipped_results + batch_results)

        return final_scores

    def export_top_poses(
        self,
        n: int = 10,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Export the top-N docked poses as a single SDF file.
        Uses DockingAnalyzer (meeko) to extract real docked coordinates from DLG/PDBQT.

        Args:
            n: Number of top molecules to export.
            output_path: Path for the SDF. Defaults to run_dir/top_{n}_poses.sdf.

        Returns:
            Path to the written SDF file.
        """
        if self.results_df.is_empty():
            raise RuntimeError("No results to export.")
        if output_path is None:
            output_path = self._run_dir / f"top_{n}_poses.sdf"
        else:
            output_path = Path(output_path)

        top_df = (
            self.results_df
            .filter(pl.col("skip_reason").is_null())
            .sort("normalized_score", descending=True)
            .head(n)
        )
        self._ensure_components()
        self._docking_analyzer.save_best_poses_sdf(
            output_path=output_path,
            results_df=top_df,
            score_col="docking_score",
            dlg_col="dlg_path",
        )
        print(f"[FCGMB] Exported top {n} poses to {output_path}")
        return output_path

    def fetch_poses(self, smiles: Optional[str] = None, top_n: int = 10) -> List:
        """
        Return RDKit molecules with actual docked 3D coordinates.

        Args:
            smiles: If given, fetch poses only for this SMILES.
            top_n: Otherwise, return poses for the top-N scoring molecules.

        Returns:
            List of RDKit Mol objects with docked coordinates.
        """
        from meeko import PDBQTMolecule, RDKitMolCreate
        if self.results_df.is_empty():
            return []

        if smiles is not None:
            df = self.results_df.filter(pl.col("smiles") == smiles)
        else:
            df = (
                self.results_df
                .filter(pl.col("skip_reason").is_null())
                .sort("normalized_score", descending=True)
                .head(top_n)
            )

        mols = []
        for row in df.filter(pl.col("dlg_path").is_not_null()).iter_rows(named=True):
            pose_file = Path(row["dlg_path"])
            if not pose_file.exists():
                continue
            try:
                is_dlg = pose_file.suffix.lower() == ".dlg"
                pdbqt_mol = PDBQTMolecule.from_file(str(pose_file), is_dlg=is_dlg, skip_typing=True)
                rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
                if rdkit_mols:
                    best = rdkit_mols[0]
                    best.SetProp("SMILES", row["smiles"])
                    best.SetProp("docking_score", str(row["docking_score"]))
                    best.SetProp("normalized_score", str(row["normalized_score"]))
                    mols.append(best)
            except Exception as e:
                print(f"[FCGMB] fetch_poses: could not read {pose_file}: {e}")
        return mols

    def save_metrics(self, extra: Optional[dict] = None) -> Path:
        """
        Save timing and performance metrics to metrics.json in the run directory.

        Args:
            extra: Additional key/value pairs to include (e.g. model name, seed).

        Returns:
            Path to the written metrics file.
        """
        n_docked = len(self.results_df.filter(pl.col("skip_reason").is_null())) \
            if not self.results_df.is_empty() else 0
        n_total = len(self.results_df) if not self.results_df.is_empty() else 0

        metrics: dict = {
            "benchmark": self.benchmark_name,
            "budget_used": self.budget_used,
            "budget_total": self.max_budget,
            "generation_rounds": self._generation_round,
            "timing": {
                "total_prep_time_sec": round(self._total_prep_time, 2),
                "total_dock_time_sec": round(self._total_dock_time, 2),
                "total_analysis_time_sec": round(self._total_analysis_time, 2),
                "avg_prep_time_per_mol_sec": round(
                    self._total_prep_time / max(1, self._n_prepped), 4
                ),
                "avg_dock_time_per_mol_sec": round(
                    self._total_dock_time / max(1, self._n_docked), 4
                ),
                "avg_analysis_time_per_mol_sec": round(
                    self._total_analysis_time / max(1, self._n_docked), 4
                ),
            },
            "results_summary": {
                "n_total_scored": n_total,
                "n_valid_poses": n_docked,
            },
        }
        if extra:
            metrics.update(extra)

        metrics_path = self._run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"[FCGMB] Metrics saved to {metrics_path}")
        return metrics_path

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _print_verbose_init(self):
        sep = "═" * 44
        print(f"[FCGMB] {sep}")
        print(f"[FCGMB] Benchmark: {self.benchmark_name}")
        print(f"[FCGMB] Backend:   {self._resolved_backend.upper()}")
        print(f"[FCGMB] Hardware:  {self.n_cpus} CPUs, {self.n_gpus} GPUs")
        if self._resolved_backend == "vina":
            print(f"[FCGMB] Vina exhaustiveness: {self._backend_config['vina_exhaustiveness']}")
        print(f"[FCGMB] {sep}")

    def _get_full_data_and_threshold(self) -> Tuple[pl.DataFrame, float, str]:
        """Load bioactivity data and compute the 25 % activity threshold.

        Lookup order:
        1. In-memory cache
        2. Package-bundled CSV  (fcgmb/bioactivity_data/<name>.csv)
        3. Scratch cache        (~/.fcgmb/bioactivity_data/<name>_chembl.csv)
        4. Live ChEMBL fetch
        """
        if self._chembl_data is not None:
            df = self._chembl_data
        else:
            df = pl.DataFrame()

            pkg_file = self._pkg_bioactivity_dir / f"{self.benchmark_name}.csv"
            if pkg_file.exists():
                print(f"[FCGMB] Loading bundled bioactivity data from {pkg_file.name}")
                df = pl.read_csv(pkg_file)

            if df.is_empty():
                cache_file = self._bioactivity_data_dir / f"{self.benchmark_name}_chembl.csv"
                if cache_file.exists():
                    print(f"[FCGMB] Loading cached ChEMBL data from {cache_file.name}")
                    df = pl.read_csv(cache_file)

            if df.is_empty():
                print(f"[FCGMB] Downloading bioactivity data from ChEMBL for {self._target_id}...")
                df = fetch_chembl_data(self._target_id, self._doc_id)
                if not df.is_empty():
                    # Lazily create the cache dir before first write
                    self._bioactivity_data_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = self._bioactivity_data_dir / f"{self.benchmark_name}_chembl.csv"
                    df.write_csv(cache_file)
                    print(f"[FCGMB] Saved ChEMBL data to {cache_file}")

            if not df.is_empty():
                self._chembl_data = df

        if df.is_empty():
            print("[FCGMB] Warning: No compounds found for this benchmark.")
            return df, 0.0, ""

        if "pchembl_value" not in df.columns:
            raise RuntimeError("Missing pchembl_value in bioactivity data.")

        act_col = "pchembl_value"
        pvals = df.get_column(act_col).to_numpy()

        # Use the empirical 25th percentile as the activity threshold instead of
        # a simple linear interpolation between min and max. This matches the
        # intended "lowest-quartile bioactivity" description and yields a
        # more stable initial context size across benchmarks.
        if pvals.size == 0:
            threshold = 0.0
        else:
            threshold = float(np.quantile(pvals, 0.25))

        return df, threshold, act_col

    def _resolve_backend(self) -> str:
        import shutil

        adgpu_exe = self._backend_config.get("adgpu_executable", "adgpu")
        adgpu_ok = shutil.which(adgpu_exe) is not None or (
            Path(adgpu_exe).exists() and os.access(adgpu_exe, os.X_OK)
        )
        requested = self._docking_backend.lower()

        if requested == "autodock_gpu":
            if adgpu_ok:
                return "autodock_gpu"
            print(f"[FCGMB] Warning: AutoDock-GPU not found ('{adgpu_exe}'). Falling back to Vina.")
            return "vina"
        if requested == "vina":
            return "vina"
        # "auto"
        return "autodock_gpu" if (adgpu_ok and self.n_gpus > 0) else "vina"

    def _ensure_components(self):
        """Lazily initialise docking components and locate/build grid maps."""
        if self._docking_oracle is not None:
            return

        self._resolved_backend = self._resolve_backend()
        self._print_verbose_init()

        # Locate grid: package-bundled first, then scratch
        fld_files = []
        pkg_grid_dir = self._pkg_grids_dir / self.pdb_id
        if pkg_grid_dir.exists():
            fld_files = list(pkg_grid_dir.glob("*.maps.fld"))
            if fld_files:
                self._grid_dir = pkg_grid_dir
                print(f"[FCGMB] Using pre-built package grids for {self.pdb_id}")

        if not fld_files:
            fld_files = list(self._grid_dir.glob("*.maps.fld"))
            if fld_files:
                print(f"[FCGMB] Using scratch grids for {self.pdb_id}")

        if not fld_files:
            print(f"[FCGMB] No pre-built grid found — preparing receptor for {self.pdb_id}...")
            try:
                from .receptor import ReceptorPreparer
            except ImportError as e:
                raise ImportError(
                    "Receptor preparation requires 'prody', which is an optional dependency. "
                    "Install it with: pip install fcgmb[receptor]  or  conda install -c conda-forge prody"
                ) from e
            # Lazily create the scratch grid dir before first write
            self._grid_dir.mkdir(parents=True, exist_ok=True)
            preparer = ReceptorPreparer(
                autogrid_executable="autogrid4",
                mk_prepare_receptor_executable="mk_prepare_receptor.py",
                reduce2_executable="mmtbx.reduce2",
            )
            fld_path = preparer.prepare_receptor_and_grid(
                self.pdb_id,
                ligand_resname=self._ligand_resname,
                output_dir=self._grids_base_dir / self.pdb_id,
                allow_bad_res=True,
            )
        else:
            fld_path = fld_files[0]
            print(f"[FCGMB] Using existing grids for {self.pdb_id}: {fld_path.name}")

        # Analyzer
        ref_path = self._grid_dir / f"{self.pdb_id}_ligand_corrected.sdf"
        if not ref_path.exists():
            ref_path = self._grid_dir / f"{self.pdb_id}_ligand.pdb"
        self._docking_analyzer = DockingAnalyzer(
            reference_ligand_path=ref_path if ref_path.exists() else None,
            fragment_smiles=self._fragment_smiles,
            rmsd_threshold=self._rmsd_threshold,
        )

        # Preparer
        self._ligand_preparer = LigandPreparer(n_cpus=self.n_cpus)

        # Docking oracle
        if self._resolved_backend == "autodock_gpu":
            self._docking_oracle = AutoDockGPUOracle(
                receptor_file=fld_path,
                adgpu_executable=self._backend_config["adgpu_executable"],
                save_dir=self._results_dir,
                n_poses=self._backend_config["n_poses"],
                n_cpus=self.n_cpus,
                n_gpus=self.n_gpus,
            )
        else:
            self._docking_oracle = AutoDockVinaOracle(
                receptor_file=fld_path,
                exhaustiveness=self._backend_config["vina_exhaustiveness"],
                save_dir=self._results_dir,
                n_poses=self._backend_config["n_poses"],
                n_cpus=self.n_cpus,
            )

    def _update_results_df(self, new_results: List[Dict]):
        """
        Append new results to the in-memory results DataFrame.

        We enforce a stable schema here so that Polars does not infer mismatched
        dtypes (e.g. `Null` for columns that are initially all None) across
        different batches, which would otherwise cause SchemaError on concat.
        """
        if not new_results:
            return

        # Stable schema for all result batches
        schema = {
            "smiles": pl.Utf8,
            "docking_score": pl.Float64,
            "normalized_score": pl.Float64,
            "valid_pose_found": pl.Boolean,
            "dlg_path": pl.Utf8,
            "best_any_score": pl.Float64,
            "skip_reason": pl.Utf8,
            "n_conformers": pl.Int64,
        }

        new_df = pl.DataFrame(new_results, schema=schema)

        if self.results_df.is_empty():
            self.results_df = new_df
        else:
            # Ensure existing frame matches the same schema; cast when necessary.
            self.results_df = self.results_df.cast(schema, strict=False)
            self.results_df = pl.concat([self.results_df, new_df])

        self._flush_live_results()

    def _flush_live_results(self):
        """Write a live-updating CSV and JSON status file to the run directory."""
        if self.results_df.is_empty():
            return

        docked_df = (
            self.results_df
            .filter(pl.col("skip_reason").is_null())
            .sort("normalized_score", descending=True)
        )

        live_csv = self._run_dir / f"current_results_{self.benchmark_name}.csv"
        docked_df.write_csv(live_csv)

        n_total = len(self.results_df)
        n_docked = len(docked_df)
        best_score = float(docked_df["normalized_score"].max()) if n_docked > 0 else 0.0
        best_docking = float(docked_df["docking_score"].min()) if n_docked > 0 else float("nan")

        status = {
            "benchmark": self.benchmark_name,
            "budget_used": self.budget_used,
            "budget_total": self.max_budget,
            "generation_round": self._generation_round,
            "n_molecules_docked": n_docked,
            "n_molecules_skipped_2d": n_total - n_docked,
            "best_normalized_score": best_score,
            "best_docking_score_kcal": best_docking,
        }
        live_status = self._run_dir / f"current_status_{self.benchmark_name}.json"
        with open(live_status, "w") as f:
            json.dump(status, f, indent=2, default=str)
