# src/mockdock/loader.py
"""
Shared loader for mockdock benchmark config and bioactivity data.
Used by both MDOracle and MDEvaluator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .data import fetch_chembl_data


class BenchmarkLoader:
    """
    Lightweight loader: reads the benchmark TOML config and bioactivity CSV.
    Does NOT initialise any docking engine or create run directories.
    """

    def __init__(
        self,
        benchmark_name: str,
        scratch_dir: Path | None = None,
    ):
        self.benchmark_name = benchmark_name
        _pkg = Path(__file__).parent
        self._pkg_bioactivity_dir = _pkg / "bioactivity_data"
        _pkg_configs_dir = _pkg / "configs"
        _scratch = Path(scratch_dir).resolve() if scratch_dir else Path.home() / ".mockdock"
        self._bioactivity_data_dir = _scratch / "bioactivity_data"

        # Load config
        config_path = self._find_config(benchmark_name, _pkg_configs_dir)
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

        self.pdb_id: str = raw["pdb_id"]
        self.target_id: str = raw.get("target_id", "")
        self.doc_id: str | None = raw.get("doc_id")
        self.fragment_smiles: str = raw["fragment_smiles"]
        self.fragment_smiles_with_dummies: str | None = raw.get("fragment_smiles_with_dummies")
        self.libinvent_scaffold_with_dummies: str | None = raw.get(
            "libinvent_scaffold_with_dummies"
        )
        self.rmsd_threshold: float = raw.get("rmsd_threshold", 2.0)
        self.low_score: float | None = raw.get("low_score")
        self.high_score: float | None = raw.get("high_score")
        self.ligand_resname: str | None = raw.get("ligand_resname")
        self.require_fragment_match: bool = raw.get("require_fragment_match", True)
        self.require_pose_rmsd: bool = raw.get("require_pose_rmsd", True)
        self.filter_during_optimization: bool = raw.get("filter_during_optimization", True)
        self.clip_reward_upper_bound: bool = raw.get("clip_reward_upper_bound", True)

        self._chembl_data: pl.DataFrame | None = None
        self._threshold: float | None = None

    # ── Config helpers ────────────────────────────────────────────────

    @staticmethod
    def _find_config(name: str, config_dir: Path) -> Path:
        # Try exact, then uppercase, then lowercase
        stems = [name, name.upper(), name.lower()]
        for stem in stems:
            p = config_dir / f"{stem}.toml"
            if p.exists():
                return p

        # Fallback to local configs/ directory if not in package
        local_config_dir = Path("configs")
        if local_config_dir.exists():
            for stem in stems:
                p = local_config_dir / f"{stem}.toml"
                if p.exists():
                    return p

        available = [f.stem for f in config_dir.glob("*.toml")]
        raise FileNotFoundError(f"Benchmark config '{name}' not found. Available: {available}")

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        pkg_config_dir = Path(__file__).parent / "configs"
        if not pkg_config_dir.exists():
            return []
        return sorted(f.stem for f in pkg_config_dir.glob("*.toml"))

    # ── Bioactivity helpers ───────────────────────────────────────────

    def get_full_data_and_threshold(self) -> tuple[pl.DataFrame, float, str]:
        """
        Load bioactivity data and compute the 25th-percentile activity threshold.

        Lookup order:
        1. In-memory cache
        2. Package-bundled CSV (mockdock/bioactivity_data/<name>.csv)
        3. Scratch cache (~/.mockdock/bioactivity_data/<name>_chembl.csv)
        4. Live ChEMBL fetch
        """
        if self._chembl_data is not None:
            return self._chembl_data, self._threshold, "pchembl_value"

        df = pl.DataFrame()

        pkg_file = self._pkg_bioactivity_dir / f"{self.benchmark_name}.csv"
        if pkg_file.exists():
            df = pl.read_csv(pkg_file)

        if df.is_empty():
            cache_file = self._bioactivity_data_dir / f"{self.benchmark_name}_chembl.csv"
            if cache_file.exists():
                df = pl.read_csv(cache_file)

        if df.is_empty():
            df = fetch_chembl_data(self.target_id, self.doc_id)
            if not df.is_empty():
                self._bioactivity_data_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._bioactivity_data_dir / f"{self.benchmark_name}_chembl.csv"
                df.write_csv(cache_file)

        if df.is_empty():
            return df, 0.0, ""

        act_col = "pchembl_value"
        pvals = df.get_column(act_col).to_numpy()

        # Use the empirical 25th percentile as the activity threshold (lower bioactivity)
        threshold = float(np.quantile(pvals, 0.25)) if pvals.size > 0 else 0.0

        self._chembl_data = df
        self._threshold = threshold
        return df, threshold, act_col

    def get_initial_compounds(self) -> pl.DataFrame:
        """Return the model-visible initial compound set (lowest-quartile bioactivity)."""
        df, threshold, act_col = self.get_full_data_and_threshold()
        if df.is_empty():
            return df
        return df.filter(pl.col(act_col) <= threshold)

    def get_validation_compounds(self) -> pl.DataFrame:
        """Return the validation compound set (above the 25th-percentile threshold)."""
        df, threshold, act_col = self.get_full_data_and_threshold()
        if df.is_empty():
            return df
        return df.filter(pl.col(act_col) > threshold)
