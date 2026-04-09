# src/mockdock/evaluator.py
"""
MDEvaluator — post-hoc metric computation for a single mockdock benchmark run.

Usage (CLI):
    python -m mockdock.evaluator results.csv --benchmark CHK1 [--output eval_metrics.json]

Usage (Python API):
    from mockdock import MDEvaluator
    ev = MDEvaluator("CHK1")
    metrics = ev.compute_metrics(Path("results.csv"))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

from .loader import BenchmarkLoader

METRIC_DESCRIPTIONS = {
    "validity":                 "Fraction of generated SMILES that parse into valid RDKit molecules.",
    "uniqueness":               "Fraction of valid molecules that are structurally distinct.",
    "internal_diversity":       "Average pairwise Tanimoto distance among unique valid molecules.",
    "scaffold_diversity":       "Fraction of unique Murcko scaffolds among unique valid molecules.",
    "mean_qed":                 "Mean quantitative estimate of drug-likeness (QED) of valid molecules.",
    "mean_sa":                  "Mean synthetic accessibility score of valid molecules (1=easy, 10=hard).",
    "fragment_incorporation":   "Fraction of unique valid molecules containing the required fragment substructure.",
    "fraction_lipinski":        "Fraction of unique valid molecules passing all four Lipinski Ro5 criteria.",
    "fraction_pains_free":      "Fraction of unique valid molecules not flagging any PAINS substructure.",
    "novelty":                  "Fraction of unique valid molecules not in the model-visible initial compounds.",
    "nonidenticality":          "Fraction of valid generated molecules that are not identical to their original input molecules.",
    "effective_novelty":        "Fraction of valid generated molecules that are both novel and non-identical to their original input molecules.",
    "snn":                      "Avg max Tanimoto similarity between generated and initial compounds.",
    "mean_qed_novel":           "Mean QED of unique valid molecules that are novel against the model-visible set.",
    "mean_sa_novel":            "Mean synthetic accessibility score of unique valid molecules that are novel against the model-visible set.",
    "avg_top_1":                "Normalized docking score of the single best-scoring molecule.",
    "avg_top_10":               "Mean normalized docking score of the 10 best-scoring molecules.",
    "avg_top_100":              "Mean normalized docking score of the 100 best-scoring molecules.",
    "auc_top_10":               "AUC of running top-10 avg score curve over cumulative oracle calls.",
    "valid_pose_rate":          "Fraction of docked molecules with a pose within the RMSD threshold.",
    "oracle_efficiency_80":     "Oracle calls to reach 80% of final top-10 score (fewer is better).",
}


class MDEvaluator:
    """
    Post-hoc evaluator for a single mockdock benchmark run.

    Delegates all config/bioactivity loading to BenchmarkLoader — no docking engine
    is instantiated, making this lightweight and safe to run on a login node.
    """

    def __init__(self, benchmark_name: str, scratch_dir: Path | None = None):
        self.benchmark_name = benchmark_name
        self._loader = BenchmarkLoader(benchmark_name, scratch_dir=scratch_dir)
        self._pains_catalog = self._build_pains_catalog()

    def compute_metrics(self, results_csv: Path, output_path: Path | None = None) -> dict:
        """Compute all metrics for one results.csv file and optionally write JSON."""
        if not results_csv.exists():
            raise FileNotFoundError(f"Results CSV not found: {results_csv}")

        df = pl.read_csv(results_csv)
        ref_smiles_df = self._loader.get_initial_compounds()
        # ChEMBL data uses 'canonical_smiles'
        ref_smiles = set()
        if not ref_smiles_df.is_empty():
            col = "canonical_smiles" if "canonical_smiles" in ref_smiles_df.columns else "smiles"
            ref_smiles = set(ref_smiles_df[col].to_list())
        ref_smiles_canonical = self._canonicalize_smiles_set(ref_smiles)
        
        fragment_smiles = self._loader.fragment_smiles

        print(f"[{self.benchmark_name}] Computing metrics for {results_csv.name}...")

        # ─── Intrinsic Metrics ────────────────────────────────────────────────
        smiles_col = "smiles" if "smiles" in df.columns else "original_smiles"
        original_col = "original_smiles" if "original_smiles" in df.columns else smiles_col
        generated_raw = df[smiles_col].to_list()
        original_raw = df[original_col].to_list()

        valid_smiles = []
        valid_pairs: list[tuple[str, str]] = []
        for generated_s, original_s in zip(generated_raw, original_raw):
            gen_mol = Chem.MolFromSmiles(str(generated_s))
            if gen_mol is None:
                continue
            generated_canonical = Chem.MolToSmiles(gen_mol)
            original_canonical = self._canonicalize_smiles(str(original_s))
            if original_canonical is None:
                original_canonical = str(original_s)
            valid_smiles.append(generated_canonical)
            valid_pairs.append((generated_canonical, original_canonical))

        unique_smiles = list(set(valid_smiles))
        unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]

        metrics: dict = {}
        metrics["validity"] = len(valid_smiles) / max(len(generated_raw), 1)
        metrics["uniqueness"] = len(unique_smiles) / max(len(valid_smiles), 1)
        metrics["internal_diversity"] = self._tanimoto_diversity(unique_smiles)
        metrics["scaffold_diversity"] = self._scaffold_diversity(unique_smiles)
        metrics["mean_qed"] = float(np.mean([QED.qed(m) for m in unique_mols])) if unique_mols else 0.0
        metrics["mean_sa"] = float(np.mean([self._sa_score(m) for m in unique_mols])) if unique_mols else 0.0
        metrics["fragment_incorporation"] = self._fragment_rate(unique_smiles, fragment_smiles)
        metrics["fraction_lipinski"] = self._lipinski_fraction(unique_mols)
        metrics["fraction_pains_free"] = self._pains_free_fraction(unique_mols)

        # ─── Extrinsic Metrics ────────────────────────────────────────────────
        metrics["novelty"] = self._novelty(unique_smiles, ref_smiles_canonical)
        metrics["nonidenticality"] = (
            sum(1 for generated_canonical, original_canonical in valid_pairs if generated_canonical != original_canonical)
            / max(len(valid_pairs), 1)
        )
        metrics["effective_novelty"] = (
            sum(
                1
                for generated_canonical, original_canonical in valid_pairs
                if (generated_canonical not in ref_smiles_canonical) and (generated_canonical != original_canonical)
            )
            / max(len(valid_pairs), 1)
        )
        metrics["snn"] = self._snn(unique_smiles, list(ref_smiles_canonical))
        novel_unique_smiles = [s for s in unique_smiles if s not in ref_smiles_canonical]
        novel_unique_mols = [Chem.MolFromSmiles(s) for s in novel_unique_smiles]
        metrics["mean_qed_novel"] = (
            float(np.mean([QED.qed(m) for m in novel_unique_mols])) if novel_unique_mols else 0.0
        )
        metrics["mean_sa_novel"] = (
            float(np.mean([self._sa_score(m) for m in novel_unique_mols])) if novel_unique_mols else 0.0
        )

        # ─── Docking Performance ──────────────────────────────────────────────
        scored_df = df.filter(pl.col("skip_reason").is_null())
        if not scored_df.is_empty():
            # Normalized score: high is good (reward)
            top = scored_df.sort("normalized_score", descending=True)
            metrics["avg_top_1"] = float(top.head(1)["normalized_score"].mean())
            metrics["avg_top_10"] = float(top.head(10)["normalized_score"].mean())
            metrics["avg_top_100"] = float(top.head(100)["normalized_score"].mean())
            metrics["auc_top_10"] = self._auc_top_k(df, k=10)
            metrics["valid_pose_rate"] = (
                len(scored_df.filter(pl.col("valid_pose_found"))) / len(scored_df)
            )
            metrics["oracle_efficiency_80"] = self._oracle_efficiency(df, k=10, frac=0.80)
        else:
            metrics["avg_top_1"] = 0.0
            metrics["avg_top_10"] = 0.0
            metrics["avg_top_100"] = 0.0
            metrics["auc_top_10"] = 0.0
            metrics["valid_pose_rate"] = 0.0
            metrics["oracle_efficiency_80"] = float(len(df))

        metrics["descriptions"] = METRIC_DESCRIPTIONS
        
        if output_path is None:
            output_path = Path(results_csv).parent / "eval_metrics.json"
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    # ── helpers ───────────────────────────────────────────────────────

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
    def _build_pains_catalog() -> FilterCatalog:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog(params)

    @staticmethod
    def _tanimoto_diversity(smiles_list: list[str]) -> float:
        if len(smiles_list) < 2:
            return 0.0
        from rdkit import DataStructs
        
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols if m is not None]
        
        if len(fps) < 2:
            return 0.0
            
        n = len(fps)
        similarities = []
        # Sample if too many to avoid N^2 explosion (e.g. max 1000 molecules)
        if n > 1000:
            indices = np.random.choice(n, 1000, replace=False)
            fps = [fps[i] for i in indices]
            n = 1000
            
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        
        return float(1.0 - np.mean(similarities))

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
    def _sa_score(mol: Chem.Mol) -> float:
        try:
            # Check if sascorer is easily available (e.g. from rdkit.Contrib)
            try:
                import os
                import sys

                from rdkit.Chem import RDConfig
                
                contrib_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
                if contrib_path not in sys.path:
                    sys.path.append(contrib_path)
                    
                import sascorer
            except (ImportError, AttributeError):
                # Fallback to direct import if it's already in path
                import sascorer
                
            return sascorer.calculateScore(mol)
        except (ImportError, Exception):
            return 0.0

    @staticmethod
    def _fragment_rate(unique_smiles: list[str], fragment_smiles: str) -> float:
        if not fragment_smiles or not unique_smiles:
            return 0.0
        frag = Chem.MolFromSmarts(fragment_smiles)
        if frag is None:
            frag = Chem.MolFromSmiles(fragment_smiles)
        if frag is None:
            return 0.0
            
        hits = 0
        for s in unique_smiles:
            m = Chem.MolFromSmiles(s)
            if m and m.HasSubstructMatch(frag):
                hits += 1
        return hits / len(unique_smiles)

    @staticmethod
    def _lipinski_fraction(mols: list[Chem.Mol]) -> float:
        if not mols:
            return 0.0
        def passes_ro5(m):
            return (
                Descriptors.MolWt(m) <= 500 and
                Descriptors.NumHDonors(m) <= 5 and
                Descriptors.NumHAcceptors(m) <= 10 and
                Descriptors.MolLogP(m) <= 5
            )
        count = sum(1 for m in mols if passes_ro5(m))
        return count / len(mols)

    def _pains_free_fraction(self, mols: list[Chem.Mol]) -> float:
        if not mols:
            return 0.0
        count = sum(1 for m in mols if not self._pains_catalog.HasMatch(m))
        return count / len(mols)

    @staticmethod
    def _novelty(generated: list[str], reference: set[str]) -> float:
        if not generated:
            return 0.0
        if not reference:
            return 1.0 # Everything is novel if reference is empty
        novel = sum(1 for s in generated if s not in reference)
        return novel / len(generated)

    @staticmethod
    def _snn(generated: list[str], reference: list[str]) -> float:
        if not generated or not reference:
            return 0.0
        from rdkit import DataStructs
        
        gen_mols = [Chem.MolFromSmiles(s) for s in generated]
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in gen_mols if m is not None]
        
        ref_mols = [Chem.MolFromSmiles(s) for s in reference]
        ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in ref_mols if m is not None]
        
        if not gen_fps or not ref_fps:
            return 0.0
            
        max_sims = []
        for g_fp in gen_fps:
            sims = DataStructs.BulkTanimotoSimilarity(g_fp, ref_fps)
            max_sims.append(max(sims))
            
        return float(np.mean(max_sims))

    @staticmethod
    def _auc_top_k(df: pl.DataFrame, k: int = 10) -> float:
        """
        Compute AUC of the running top-k average normalized score.
        X-axis = cumulative oracle calls (row order).
        """
        scores = df["normalized_score"].to_list()
        running_max_k = []
        buffer = []
        
        for s in scores:
            buffer.append(s)
            # running top-k average
            top_k = sorted(buffer, reverse=True)[:k]
            running_max_k.append(np.mean(top_k))
            
        # Cumulative oracle calls as X
        x = np.arange(1, len(running_max_k) + 1)
        # Normalize by X-range to get value in [0, 1] (since score is [0, 1])
        # Use np.trapezoid if available (NumPy 2.0+), fallback to deprecated np.trapz
        trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        if trap_func is None:
            raise AttributeError("Neither np.trapezoid nor np.trapz found in NumPy.")
        auc = trap_func(running_max_k, x) / len(running_max_k)
        return float(auc)

    @staticmethod
    def _oracle_efficiency(df: pl.DataFrame, k: int = 10, frac: float = 0.80) -> int:
        """Calls to reach frac of final top-k score."""
        scores = df["normalized_score"].to_list()
        if not scores:
            return 0
            
        final_top_k_val = np.mean(sorted(scores, reverse=True)[:k])
        target = frac * final_top_k_val
        if target <= 0:
            return 1
            
        buffer = []
        for i, s in enumerate(scores):
            buffer.append(s)
            current_top_k = np.mean(sorted(buffer, reverse=True)[:k])
            if current_top_k >= target:
                return i + 1
        return len(scores)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mockdock Evaluator CLI")
    parser.add_argument("results_csv", type=Path, help="Path to results.csv")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., CHK1)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--scratch-dir", type=Path, default=None, help="mockdock scratch directory")
    
    args = parser.parse_args()
    
    evaluator = MDEvaluator(args.benchmark, scratch_dir=args.scratch_dir)
    evaluator.compute_metrics(args.results_csv, output_path=args.output)
    print(f"Metrics saved for {args.benchmark}")


if __name__ == "__main__":
    main()
