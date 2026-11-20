# src/mockdock/evaluator.py
"""
MDEvaluator — post-hoc metric computation for a single mockdock benchmark run.

Usage (CLI):
    python -m mockdock.evaluator ./my_run [--budget 500] [--output eval_metrics.json]

Usage (Python API):
    from mockdock import MDEvaluator
    ev = MDEvaluator("./my_run")
    metrics = ev.compute_metrics()
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .filters import MDFilters
from .loader import BenchmarkLoader
from .utils import get_robust_match

_SASCORE_MODULE = None


def _get_sascorer():
    global _SASCORE_MODULE
    if _SASCORE_MODULE is not None:
        return _SASCORE_MODULE
    try:
        import sascorer

        _SASCORE_MODULE = sascorer
        return sascorer
    except (ImportError, AttributeError):
        pass
    try:
        from rdkit.Chem import RDConfig

        contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib_path not in sys.path:
            sys.path.append(contrib_path)
        import sascorer

        _SASCORE_MODULE = sascorer
        return sascorer
    except Exception:
        return None


METRIC_DESCRIPTIONS = {
    "validity": "Fraction of generated SMILES that parse into valid RDKit molecules.",
    "uniqueness": "Fraction of valid molecules that are structurally distinct.",
    "internal_diversity": "Average pairwise Tanimoto distance among unique valid molecules.",
    "scaffold_diversity": "Fraction of unique Murcko scaffolds among unique valid molecules.",
    "mean_qed": "Mean quantitative estimate of drug-likeness (QED) of unique valid molecules.",
    "mean_sa": "Mean synthetic accessibility score of unique valid molecules (1=easy, 10=hard; null if sascorer unavailable).",
    "fragment_adherence": "Fraction of unique valid molecules containing the required 2D fragment.",
    "novelty": "Fraction of unique valid molecules not in the model-visible initial compounds.",
    "snn": "Avg max Tanimoto similarity between unique valid molecules and initial compounds.",
    "fraction_medchem_pass": "Fraction of unique valid fragment-containing molecules passing PAINS/BMS alerts and physchem bounds.",
    "effective_yield_rate": "Fraction of all generated molecules that are valid, unique, novel, and contain the 2D fragment.",
    "effective_medchem_pass": "Fraction of Effective Yield Rate molecules passing PAINS/BMS alerts and physchem bounds.",
    "avg_top_1": "Reward score of the single best-scoring molecule (zero-padded; null if budget < 1).",
    "avg_top_10": "Mean reward score of the 10 best-scoring unique molecules (zero-padded; null if budget < 10).",
    "avg_top_100": "Mean reward score of the 100 best-scoring unique molecules (zero-padded; null if budget < 100).",
    "auc_top_10": "AUC of running top-10 reward curve over cumulative oracle calls for unique molecules (trapezoidal integration over 1..budget divided by budget - 1; carries final score forward for early-stopped runs).",
    "valid_pose_rate": "Fraction of docked molecules with a pose within the RMSD threshold.",
}


class MDEvaluator:
    """
    Post-hoc evaluator for a single mockdock benchmark run.

    Reads benchmark metadata and execution logs directly from an MDOracle run directory.
    Delegates config and bioactivity loading to BenchmarkLoader without instantiating
    a docking engine, making it safe and fast to run on login nodes.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        budget: int | None = None,
        scratch_dir: Path | str | None = None,
        max_tanimoto_samples: int = 1000,
        seed: int = 42,
    ):
        target_path = Path(run_dir).resolve()
        # Forgiving path resolution: if user passes results.csv or any file inside run_dir
        if target_path.is_file():
            if (target_path.parent / "status.json").exists():
                target_path = target_path.parent

        status_file = target_path / "status.json"
        if not status_file.exists():
            raise FileNotFoundError(
                f"Run directory '{target_path}' does not contain 'status.json'. "
                f"An MDOracle run directory containing status.json is required. "
                f"If you are evaluating an external CSV, use MDEvaluator.from_csv(...) instead."
            )

        with open(status_file) as f:
            self._status = json.load(f)

        self.run_dir: Path | None = target_path
        self._csv_path: Path | None = target_path / "results.csv"

        self.benchmark_name: str = self._status.get("benchmark", "")
        if not self.benchmark_name:
            raise ValueError(f"Invalid status.json in '{target_path}': missing 'benchmark' field.")

        self._model: str | None = self._status.get("model")
        self.budget_total: int | None = self._status.get("budget_total")

        if budget is not None:
            if budget <= 0:
                raise ValueError(f"budget must be a positive integer, got {budget}")
            self.budget: int | None = budget
        else:
            self.budget = None

        self.scratch_dir: Path | None = (
            Path(scratch_dir).resolve() if scratch_dir is not None else None
        )
        self._loader = BenchmarkLoader(self.benchmark_name, scratch_dir=self.scratch_dir)
        self._filters = MDFilters(active_rulesets=["PAINS", "BMS"])

        # Prioritize fragment_smiles from status.json
        if self._status.get("fragment_smiles"):
            self.fragment_smiles: str = self._status["fragment_smiles"]
        elif hasattr(self._loader, "get_fragment_smiles"):
            self.fragment_smiles = self._loader.get_fragment_smiles(self._model)
        else:
            self.fragment_smiles = getattr(self._loader, "fragment_smiles", "")

        self.max_tanimoto_samples: int = max_tanimoto_samples
        self.seed: int = seed

    @classmethod
    def from_oracle(cls, oracle, **kwargs) -> MDEvaluator:
        """Instantiate MDEvaluator directly from a live MDOracle session."""
        return cls(oracle.run_dir, **kwargs)

    @classmethod
    def from_csv(
        cls,
        results_csv: str | Path | None,
        benchmark: str,
        *,
        budget: int | None = None,
        model: str | None = None,
        fragment_smiles: str | None = None,
        scratch_dir: Path | str | None = None,
        max_tanimoto_samples: int = 1000,
        seed: int = 42,
    ) -> MDEvaluator:
        """Instantiate MDEvaluator for an external CSV without requiring status.json."""
        if results_csv is not None:
            csv_path = Path(results_csv).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"Results CSV not found: {csv_path}")
        else:
            csv_path = None

        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be a positive integer, got {budget}")

        obj = cls.__new__(cls)
        obj.run_dir = None
        obj._status = {}
        obj._csv_path = csv_path
        obj.benchmark_name = benchmark
        obj._model = model
        obj.budget_total = budget
        obj.budget = budget
        obj.scratch_dir = Path(scratch_dir).resolve() if scratch_dir is not None else None
        obj._loader = BenchmarkLoader(benchmark, scratch_dir=obj.scratch_dir)
        obj._filters = MDFilters(active_rulesets=["PAINS", "BMS"])

        if fragment_smiles is not None:
            obj.fragment_smiles = fragment_smiles
        elif hasattr(obj._loader, "get_fragment_smiles"):
            obj.fragment_smiles = obj._loader.get_fragment_smiles(model)
        else:
            obj.fragment_smiles = getattr(obj._loader, "fragment_smiles", "")

        obj.max_tanimoto_samples = max_tanimoto_samples
        obj.seed = seed
        return obj

    @classmethod
    def compute_reference_metrics(
        cls, benchmark: str, *, scratch_dir: Path | str | None = None
    ) -> pl.DataFrame:
        """
        Return baseline reference compound properties (smiles, qed, sa) for a benchmark.
        Safe to run on login nodes; requires no run directory or active docking session.
        """
        loader = BenchmarkLoader(benchmark, scratch_dir=scratch_dir)
        compounds_df = loader.get_initial_compounds()
        if compounds_df.is_empty():
            compounds_df = loader.get_validation_compounds()

        if compounds_df.is_empty():
            return pl.DataFrame(schema={"smiles": pl.Utf8, "qed": pl.Float64, "sa": pl.Float64})

        col = "canonical_smiles" if "canonical_smiles" in compounds_df.columns else "smiles"
        raw_smiles = compounds_df[col].to_list()

        rows = []
        for s in raw_smiles:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            q = float(QED.qed(mol))
            sa = cls._sa_score(mol)
            rows.append({"smiles": can, "qed": q, "sa": sa})

        if not rows:
            return pl.DataFrame(schema={"smiles": pl.Utf8, "qed": pl.Float64, "sa": pl.Float64})

        return pl.DataFrame(rows).unique(subset=["smiles"], keep="first")

    def load_results(self, budget: int | None = None) -> pl.DataFrame:
        """
        Load and filter results from results.csv.

        Filters out uncounted / budget_exhausted rows and slices to effective_budget.
        """
        if self._csv_path is None or not self._csv_path.exists():
            dir_str = f" in {self.run_dir}" if self.run_dir else ""
            raise FileNotFoundError(
                f"No results.csv found{dir_str}. Has oracle.score() been run yet?"
            )

        df = pl.read_csv(self._csv_path)

        # In-budget row filtering:
        # If oracle_call exists, keep rows where oracle_call is not null.
        # Otherwise filter out budget_exhausted rows.
        if "oracle_call" in df.columns:
            df = df.filter(pl.col("oracle_call").is_not_null())
        elif "skip_reason" in df.columns:
            df = df.filter(
                pl.col("skip_reason").is_null() | (pl.col("skip_reason") != "budget_exhausted")
            )

        effective_budget = budget if budget is not None else self.budget
        if effective_budget is not None:
            if effective_budget <= 0:
                raise ValueError(f"budget must be a positive integer, got {effective_budget}")
            df = df.head(effective_budget)

        return df

    def compute_metrics(
        self,
        budget: int | None = None,
        output_path: Path | str | None = None,
        save: bool | None = None,
    ) -> dict:
        """
        Compute all benchmark metrics for this run.

        Args:
            budget: Optional budget cutoff override.
            output_path: Optional explicit output JSON path.
            save: Whether to save eval_metrics.json. If None, auto-saves to run_dir
                only when evaluating the full run budget (effective_budget == budget_total).
        """
        effective_budget = budget if budget is not None else self.budget
        df = self.load_results(budget=effective_budget)

        ref_smiles_df = self._loader.get_initial_compounds()
        ref_smiles = set()
        if not ref_smiles_df.is_empty():
            col = "canonical_smiles" if "canonical_smiles" in ref_smiles_df.columns else "smiles"
            ref_smiles = set(ref_smiles_df[col].to_list())
        ref_smiles_canonical = self._canonicalize_smiles_set(ref_smiles)

        smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
        generated_raw = (
            df[smiles_col].to_list() if not df.is_empty() and smiles_col in df.columns else []
        )

        valid_smiles = []
        mol_map: dict[str, Chem.Mol] = {}
        for generated_s in generated_raw:
            gen_mol = Chem.MolFromSmiles(str(generated_s))
            if gen_mol is None:
                continue
            generated_canonical = Chem.MolToSmiles(gen_mol)
            if generated_canonical not in mol_map:
                mol_map[generated_canonical] = gen_mol
            valid_smiles.append(generated_canonical)

        unique_smiles = list(mol_map.keys())
        unique_mols = [mol_map[s] for s in unique_smiles]

        metrics: dict = {
            "benchmark": self.benchmark_name,
            "model": self._model,
            "budget_total": self.budget_total,
            "evaluated_budget": effective_budget,
            "n_evaluated_rows": len(df),
        }

        # ─── Generation Metrics ──────────────────────────────────────────────
        metrics["validity"] = len(valid_smiles) / max(len(generated_raw), 1)
        metrics["uniqueness"] = len(unique_smiles) / max(len(valid_smiles), 1)
        metrics["internal_diversity"] = self._tanimoto_diversity(
            unique_smiles, max_samples=self.max_tanimoto_samples, seed=self.seed
        )
        metrics["scaffold_diversity"] = self._scaffold_diversity(unique_smiles)

        metrics["mean_qed"] = (
            float(np.mean([QED.qed(m) for m in unique_mols])) if unique_mols else 0.0
        )

        sa_scores = [self._sa_score(m) for m in unique_mols]
        if not unique_mols:
            metrics["mean_sa"] = 0.0
        elif any(s is None for s in sa_scores):
            metrics["mean_sa"] = None
        else:
            metrics["mean_sa"] = float(np.mean(sa_scores))

        frag_passing_set = self._fragment_passing_set(unique_smiles, self.fragment_smiles)
        metrics["fragment_adherence"] = len(frag_passing_set) / max(len(unique_smiles), 1)

        # MedChem Filtering — evaluated once on all unique molecules
        filter_results = [self._filters.evaluate(m) for m in unique_mols]
        frag_pass_results = [
            r for s, r in zip(unique_smiles, filter_results) if s in frag_passing_set
        ]
        metrics["fraction_medchem_pass"] = (
            sum(1 for r in frag_pass_results if r["pass"]) / len(frag_pass_results)
            if frag_pass_results
            else 0.0
        )

        # ─── Extrinsic Metrics ────────────────────────────────────────────────
        metrics["novelty"] = self._novelty(unique_smiles, ref_smiles_canonical)
        metrics["snn"] = self._snn(unique_smiles, list(ref_smiles_canonical))
        novel_unique_smiles = [s for s in unique_smiles if s not in ref_smiles_canonical]
        novel_frag_set = {s for s in novel_unique_smiles if s in frag_passing_set}
        metrics["effective_yield_rate"] = len(novel_frag_set) / max(len(generated_raw), 1)

        novel_frag_results = [
            r for s, r in zip(unique_smiles, filter_results) if s in novel_frag_set
        ]
        metrics["effective_medchem_pass"] = (
            sum(1 for r in novel_frag_results if r["pass"]) / len(novel_frag_results)
            if novel_frag_results
            else 0.0
        )

        # ─── Docking & Optimization Metrics ──────────────────────────────────
        if not df.is_empty():
            if "skip_reason" in df.columns:
                scored_df = df.filter(pl.col("skip_reason").is_null())
            else:
                scored_df = df

            if not scored_df.is_empty() and "reward_score" in scored_df.columns:
                unique_scored = (
                    scored_df.sort("reward_score", descending=True)
                    .unique(subset=[smiles_col], keep="first")
                    .sort("reward_score", descending=True)
                )
                top_scores = unique_scored["reward_score"].to_list()
            else:
                top_scores = []
        else:
            scored_df = pl.DataFrame()
            top_scores = []

        for k in [1, 10, 100]:
            if effective_budget is not None and effective_budget < k:
                metrics[f"avg_top_{k}"] = None
            else:
                padded = (top_scores + [0.0] * k)[:k]
                metrics[f"avg_top_{k}"] = float(sum(padded) / k)

        metrics["auc_top_10"] = self._auc_top_k(df, k=10, effective_budget=effective_budget)

        # valid_pose_rate denominator: all docked molecules (skip_reason is null or failed_rmsd)
        if not df.is_empty() and "valid_pose_found" in df.columns:
            if "skip_reason" in df.columns:
                docked_df = df.filter(
                    pl.col("skip_reason").is_null() | (pl.col("skip_reason") == "failed_rmsd")
                )
            else:
                docked_df = df

            if not docked_df.is_empty():
                metrics["valid_pose_rate"] = float(
                    len(docked_df.filter(pl.col("valid_pose_found"))) / len(docked_df)
                )
            else:
                metrics["valid_pose_rate"] = 0.0
        else:
            metrics["valid_pose_rate"] = 0.0

        metrics["descriptions"] = METRIC_DESCRIPTIONS

        # ─── Auto-Save Protection Logic ──────────────────────────────────────
        should_save = False
        if save is True:
            should_save = True
        elif save is None:
            if output_path is not None:
                should_save = True
            elif self.run_dir is not None:
                if effective_budget is None:
                    should_save = True
                elif (
                    self.budget_total is not None
                    and effective_budget == self.budget_total
                ):
                    should_save = True
                elif self.budget_total is None:
                    should_save = True

        if should_save:
            if output_path is not None:
                out_file = Path(output_path)
            elif self.run_dir is not None:
                out_file = self.run_dir / "eval_metrics.json"
            elif self._csv_path is not None:
                out_file = self._csv_path.parent / "eval_metrics.json"
            else:
                out_file = Path.cwd() / "eval_metrics.json"

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2, allow_nan=False)

        return metrics

    def compute_molecule_metrics(self, budget: int | None = None) -> pl.DataFrame:
        """
        Compute per-unique-molecule quality table for in-budget molecules.

        Returns DataFrame with columns:
            [smiles, qed, sa, is_novel, has_fragment, passes_medchem,
             best_reward_score, best_docking_score, valid_pose_found]
        """
        effective_budget = budget if budget is not None else self.budget
        df = self.load_results(budget=effective_budget)

        smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
        generated_raw = (
            df[smiles_col].to_list() if not df.is_empty() and smiles_col in df.columns else []
        )

        ref_smiles_df = self._loader.get_initial_compounds()
        ref_smiles = set()
        if not ref_smiles_df.is_empty():
            col = "canonical_smiles" if "canonical_smiles" in ref_smiles_df.columns else "smiles"
            ref_smiles = set(ref_smiles_df[col].to_list())
        ref_smiles_canonical = self._canonicalize_smiles_set(ref_smiles)

        # Track unique valid molecules in first-seen order
        mol_map: dict[str, Chem.Mol] = {}
        for s in generated_raw:
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                continue
            can = Chem.MolToSmiles(m)
            if can not in mol_map:
                mol_map[can] = m

        unique_smiles = list(mol_map.keys())
        frag_passing_set = self._fragment_passing_set(unique_smiles, self.fragment_smiles)

        # Aggregate docking results per unique molecule if columns exist
        mol_docking_stats: dict[str, dict] = {}
        if not df.is_empty() and smiles_col in df.columns:
            has_reward = "reward_score" in df.columns
            has_docking = "docking_score" in df.columns
            has_val_pose = "valid_pose_found" in df.columns

            for row in df.iter_rows(named=True):
                raw_s = row[smiles_col]
                m = Chem.MolFromSmiles(str(raw_s))
                if m is None:
                    continue
                can = Chem.MolToSmiles(m)
                if can not in mol_docking_stats:
                    mol_docking_stats[can] = {
                        "best_reward": float("-inf"),
                        "best_docking": float("inf"),
                        "valid_pose_found": False,
                    }

                stat = mol_docking_stats[can]
                if has_reward and row["reward_score"] is not None:
                    r_val = float(row["reward_score"])
                    if not np.isnan(r_val) and r_val > stat["best_reward"]:
                        stat["best_reward"] = r_val
                if has_docking and row["docking_score"] is not None:
                    d_val = float(row["docking_score"])
                    if not np.isnan(d_val) and d_val < stat["best_docking"]:
                        stat["best_docking"] = d_val
                if has_val_pose and row["valid_pose_found"] is True:
                    stat["valid_pose_found"] = True

        rows = []
        for can in unique_smiles:
            mol = mol_map[can]
            q = float(QED.qed(mol))
            sa = self._sa_score(mol)
            is_novel = can not in ref_smiles_canonical
            has_frag = can in frag_passing_set
            medchem_pass = bool(self._filters.evaluate(mol)["pass"])

            stat = mol_docking_stats.get(
                can,
                {"best_reward": float("-inf"), "best_docking": float("inf"), "valid_pose_found": False},
            )
            best_rew = stat["best_reward"] if stat["best_reward"] != float("-inf") else None
            best_dock = stat["best_docking"] if stat["best_docking"] != float("inf") else None
            val_pose = stat["valid_pose_found"]

            rows.append(
                {
                    "smiles": can,
                    "qed": q,
                    "sa": sa,
                    "is_novel": is_novel,
                    "has_fragment": has_frag,
                    "passes_medchem": medchem_pass,
                    "best_reward_score": best_rew,
                    "best_docking_score": best_dock,
                    "valid_pose_found": val_pose,
                }
            )

        schema = {
            "smiles": pl.Utf8,
            "qed": pl.Float64,
            "sa": pl.Float64,
            "is_novel": pl.Boolean,
            "has_fragment": pl.Boolean,
            "passes_medchem": pl.Boolean,
            "best_reward_score": pl.Float64,
            "best_docking_score": pl.Float64,
            "valid_pose_found": pl.Boolean,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)

    def compute_trajectory(
        self, k: int = 10, budget: int | None = None, pad_to_budget: bool = False
    ) -> pl.DataFrame:
        """
        Compute running top-k reward trajectory over cumulative oracle calls.

        Args:
            k: Top-k cutoff (default 10).
            budget: Optional budget cutoff override.
            pad_to_budget: If True and effective_budget is set, pad the curve
                with the final top-k score forward through effective_budget.
                If False (default), returns observed calls (1..n).
        """
        effective_budget = budget if budget is not None else self.budget
        df = self.load_results(budget=effective_budget)

        if df.is_empty() or "reward_score" not in df.columns:
            return pl.DataFrame(schema={"oracle_call": pl.Int64, "running_top_k": pl.Float64})

        scores = df["reward_score"].to_list()
        smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
        smiles = df[smiles_col].to_list() if smiles_col in df.columns else [None] * len(scores)

        running_max_k = []
        unique_seen: dict[str | int, float] = {}

        for idx, (s, sm) in enumerate(zip(scores, smiles)):
            key = sm if sm is not None else idx
            score_val = float(s) if s is not None and not np.isnan(s) else 0.0
            if key not in unique_seen or score_val > unique_seen[key]:
                unique_seen[key] = score_val

            top_k = (sorted(unique_seen.values(), reverse=True) + [0.0] * k)[:k]
            running_max_k.append(float(sum(top_k) / k))

        calls = list(range(1, len(running_max_k) + 1))

        if pad_to_budget and effective_budget is not None and len(running_max_k) < effective_budget:
            last_val = running_max_k[-1] if running_max_k else 0.0
            extra_calls = list(range(len(running_max_k) + 1, effective_budget + 1))
            calls.extend(extra_calls)
            running_max_k.extend([last_val] * len(extra_calls))

        return pl.DataFrame(
            {"oracle_call": calls, "running_top_k": running_max_k},
            schema={"oracle_call": pl.Int64, "running_top_k": pl.Float64},
        )

    def summary(self, metrics: dict | None = None) -> str:
        """Format metrics into a concise human-readable summary table."""
        if metrics is None:
            metrics = self.compute_metrics()

        def _fmt(val, fmt_str=".3f"):
            if val is None:
                return "N/A"
            return f"{val:{fmt_str}}"

        lines = [
            "===========================================================",
            f" mockdock Evaluation Summary: {metrics.get('benchmark', self.benchmark_name)}",
            "===========================================================",
            f" Model:               {metrics.get('model', 'N/A')}",
            f" Budget Total:        {metrics.get('budget_total', 'N/A')}",
            f" Evaluated Budget:    {metrics.get('evaluated_budget', 'N/A')}",
            f" Evaluated Rows:      {metrics.get('n_evaluated_rows', 'N/A')}",
            "-----------------------------------------------------------",
            " Generation:",
            f"   Validity:          {metrics.get('validity', 0.0):.1%}",
            f"   Uniqueness:        {metrics.get('uniqueness', 0.0):.1%}",
            f"   Novelty:           {metrics.get('novelty', 0.0):.1%}",
            f"   Fragment Adherence:{metrics.get('fragment_adherence', 0.0):.1%}",
            f"   Effective Yield:   {metrics.get('effective_yield_rate', 0.0):.1%}",
            "-----------------------------------------------------------",
            " Diversity & Chemical Quality:",
            f"   Internal Diversity:{metrics.get('internal_diversity', 0.0):.3f}",
            f"   Scaffold Diversity:{metrics.get('scaffold_diversity', 0.0):.3f}",
            f"   SNN:               {metrics.get('snn', 0.0):.3f}",
            f"   Mean QED:          {metrics.get('mean_qed', 0.0):.3f}",
            f"   Mean SA:           {_fmt(metrics.get('mean_sa'), '.2f')}",
            f"   MedChem Pass:      {metrics.get('fraction_medchem_pass', 0.0):.1%}",
            f"   Eff. MedChem Pass: {metrics.get('effective_medchem_pass', 0.0):.1%}",
            "-----------------------------------------------------------",
            " Optimization & Docking:",
            f"   Avg Top 1:         {_fmt(metrics.get('avg_top_1'))}",
            f"   Avg Top 10:        {_fmt(metrics.get('avg_top_10'))}",
            f"   Avg Top 100:       {_fmt(metrics.get('avg_top_100'))}",
            f"   AUC Top 10:        {metrics.get('auc_top_10', 0.0):.3f}",
            f"   Valid Pose Rate:   {metrics.get('valid_pose_rate', 0.0):.1%}",
            "===========================================================",
        ]
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_first_existing_column(df: pl.DataFrame, candidates: list[str]) -> str:
        for name in candidates:
            if name in df.columns:
                return name
        raise ValueError(
            f"None of the expected columns are present. Expected one of: {candidates}, got {df.columns}"
        )

    @staticmethod
    def _canonicalize_smiles(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    @staticmethod
    def _canonicalize_smiles_set(smiles_set: set[str]) -> set[str]:
        canonical = set()
        for s in smiles_set:
            c = MDEvaluator._canonicalize_smiles(s)
            if c is not None:
                canonical.add(c)
        return canonical

    @staticmethod
    def _tanimoto_diversity(
        smiles_list: list[str], max_samples: int = 1000, seed: int = 42
    ) -> float:
        if len(smiles_list) < 2:
            return 0.0

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols if m is not None
        ]

        if len(fps) < 2:
            return 0.0

        n = len(fps)
        if n > max_samples:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(n), max_samples))
            fps = [fps[i] for i in indices]
            n = len(fps)

        similarities = []
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            similarities.extend(sims)

        if not similarities:
            return 0.0
        return float(1.0 - (sum(similarities) / len(similarities)))

    @staticmethod
    def _scaffold_diversity(smiles_list: list[str]) -> float:
        if not smiles_list:
            return 0.0
        scaffolds = set()
        for s in smiles_list:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
            if scaffold:
                scaffolds.add(scaffold)
        return len(scaffolds) / len(smiles_list)

    @staticmethod
    def _sa_score(mol: Chem.Mol) -> float | None:
        try:
            scorer = _get_sascorer()
            if scorer is not None and mol is not None:
                return float(scorer.calculateScore(mol))
            if scorer is None:
                warnings.warn(
                    "sascorer module unavailable; returning None for SA score.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            return 0.0
        except Exception as e:
            warnings.warn(f"Error calculating SA score: {e}", RuntimeWarning, stacklevel=2)
            return None

    @staticmethod
    def _fragment_passing_set(unique_smiles: list[str], fragment_smiles: str) -> set[str]:
        """Return the set of SMILES containing the required fragment substructure."""
        if not fragment_smiles or not unique_smiles:
            return set()
        frag = Chem.MolFromSmiles(fragment_smiles)
        if frag is None:
            frag = Chem.MolFromSmarts(fragment_smiles)
        if frag is None:
            return set()

        passing = set()
        for s in unique_smiles:
            m = Chem.MolFromSmiles(s)
            if m and get_robust_match(m, frag):
                passing.add(s)
        return passing

    @staticmethod
    def _novelty(generated: list[str], reference: set[str]) -> float:
        if not generated:
            return 0.0
        if not reference:
            return 1.0
        novel = sum(1 for s in generated if s not in reference)
        return novel / len(generated)

    @staticmethod
    def _snn(generated: list[str], reference: list[str]) -> float:
        if not generated or not reference:
            return 0.0

        gen_mols = [Chem.MolFromSmiles(s) for s in generated]
        gen_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in gen_mols if m is not None
        ]

        ref_mols = [Chem.MolFromSmiles(s) for s in reference]
        ref_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in ref_mols if m is not None
        ]

        if not gen_fps or not ref_fps:
            return 0.0

        max_sims = []
        for g_fp in gen_fps:
            sims = DataStructs.BulkTanimotoSimilarity(g_fp, ref_fps)
            max_sims.append(max(sims))

        return float(np.mean(max_sims))

    @staticmethod
    def _auc_top_k(df: pl.DataFrame, k: int = 10, effective_budget: int | None = None) -> float:
        """
        Compute AUC of the running top-k average reward score over unique molecules.

        Trapezoidal integration over cumulative oracle calls (divided by budget - 1).
        When effective_budget is set, carries final top-k score forward to effective_budget.
        When effective_budget is None, falls back to observed rows N and (N - 1) normalization.
        """
        if df.is_empty() or "reward_score" not in df.columns:
            return 0.0

        scores = df["reward_score"].to_list()
        smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
        smiles = df[smiles_col].to_list() if smiles_col in df.columns else [None] * len(scores)

        running_max_k = []
        unique_seen: dict[str | int, float] = {}

        for idx, (s, sm) in enumerate(zip(scores, smiles)):
            key = sm if sm is not None else idx
            score_val = float(s) if s is not None and not np.isnan(s) else 0.0
            if key not in unique_seen or score_val > unique_seen[key]:
                unique_seen[key] = score_val

            # Zero-padded top-k average over unique molecules
            top_k = (sorted(unique_seen.values(), reverse=True) + [0.0] * k)[:k]
            running_max_k.append(float(sum(top_k) / k))

        # Carry final score forward to effective_budget if specified
        if effective_budget is not None and effective_budget > len(running_max_k):
            last_val = running_max_k[-1] if running_max_k else 0.0
            running_max_k.extend([last_val] * (effective_budget - len(running_max_k)))

        n_pts = len(running_max_k)
        if n_pts == 0:
            return 0.0
        if n_pts == 1:
            return float(running_max_k[0])

        # Trapezoidal area on uniform grid x = 1..n_pts (dx = 1):
        # Area = 0.5 * (y[0] + y[-1]) + sum(y[1:-1])
        # Normalized by dividing by (n_pts - 1)
        area = 0.5 * (running_max_k[0] + running_max_k[-1]) + sum(running_max_k[1:-1])
        return float(area / (n_pts - 1))


# ─── CLI Entry Point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="mockdock Evaluator CLI")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--budget", type=int, default=None, help="Evaluation budget cutoff")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument(
        "--scratch-dir", type=Path, default=None, help="mockdock scratch directory"
    )

    args = parser.parse_args()

    evaluator = MDEvaluator(args.run_dir, budget=args.budget, scratch_dir=args.scratch_dir)
    metrics = evaluator.compute_metrics(budget=args.budget, output_path=args.output)
    print(evaluator.summary(metrics))


if __name__ == "__main__":
    main()
