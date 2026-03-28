# FCGMBEvaluator Implementation Plan

## Overview
It **makes perfect sense to use the `moleval` package directly from `MolScore`**. `MolScore` (and its `moleval` subpackage) is a robust, well-tested framework originally adapted from the MOSES benchmark. Re-implementing complex metrics like Sphere Exclusion Diversity (SEDiv), Frechet ChemNet Distance (FCD), Scaffold Uniqueness, and Novelty from scratch would be redundant and error-prone. By integrating `moleval.metrics.GetMetrics`, we can combine standard 2D generative model metrics with `fcgmb`'s structure-based (docking) scoring to provide a holistic view of model performance.

## 1. Plan

1.  **Introduce `FCGMBEvaluator`**: Create a new class (`src/fcgmb/evaluator.py`) responsible for aggregating both intrinsic/extrinsic 2D metrics (via `moleval`) and 3D docking-specific metrics (from `FCGMBOracle`).
2.  **Optional Dependency Management**: Add `MolScore` as an optional dependency in `pyproject.toml` (e.g., `fcgmb[evaluate]`), since it brings heavier ML dependencies (like PyTorch for FCD) that aren't strictly required for running the docking oracle.
3.  **Define Reference Sets**:
    -   **Train Set**: The ChEMBL `initial_compounds` (lowest-quartile bioactivity) provided to the model. Used by `moleval` to calculate *Novelty*.
    -   **Test Set**: The ChEMBL `validation_compounds` (upper-quartile bioactivity). Used by `moleval` to compute *FCD* and similarity metrics against known active compounds.
4.  **Extract Generated Set**: Extract all original generated SMILES from the `FCGMBOracle.results_df`, including those skipped due to invalidity or 2D fragment mismatches, so `moleval` can correctly calculate *Validity* and *Uniqueness*.
5.  **Compute Metrics**:
    -   **Intrinsic (via moleval)**: Validity, Uniqueness, Scaffold Uniqueness, Internal Diversity (IntDiv1/IntDiv2), Sphere Exclusion Diversity (SEDiv), QED, SA score.
    -   **Extrinsic (via moleval)**: Novelty, FCD (Frechet ChemNet Distance), SNN (Nearest Neighbor Similarity).
    -   **3D/Docking (via fcgmb)**: Average Top-10 Docking Score, AUC-top10, Valid Pose Hit Rate (RMSD ≤ threshold), Fragment Match Rate.
6.  **Report Generation**: Output a consolidated JSON or Markdown report containing all metrics for easy integration into benchmarking pipelines.

## 2. Details of Implementation

```python
# src/fcgmb/evaluator.py
import polars as pl
from typing import Dict, Any, Optional

try:
    from moleval.metrics.metrics import GetMetrics
    _HAS_MOLEVAL = True
except ImportError:
    _HAS_MOLEVAL = False

class FCGMBEvaluator:
    def __init__(self, oracle: "FCGMBOracle", n_jobs: int = 1):
        if not _HAS_MOLEVAL:
            raise ImportError(
                "MolScore is required for FCGMBEvaluator. "
                "Install it with: pip install MolScore"
            )

        self.oracle = oracle
        self.n_jobs = n_jobs

        # 1. Define Reference Sets from Oracle's ChEMBL data
        self.train_df = self.oracle.get_initial_compounds()
        self.test_df = self.oracle.get_validation_compounds()

        self.train_smiles = self.train_df.get_column("smiles").to_list() if not self.train_df.is_empty() else []
        self.test_smiles = self.test_df.get_column("smiles").to_list() if not self.test_df.is_empty() else []

        # 2. Initialize moleval's GetMetrics Engine
        # We pass train/test sets here so it pre-calculates reference statistics (e.g., FCD distributions)
        self.metrics_engine = GetMetrics(
            n_jobs=self.n_jobs,
            train=self.train_smiles,
            test=self.test_smiles,
            target=None,          # Can optionally provide target-specific hits if needed
            run_fcd=True          # Enable Frechet ChemNet Distance
        )

    def evaluate(self) -> Dict[str, Any]:
        """Calculates both 2D (moleval) and 3D (docking) metrics."""
        df = self.oracle.results_df
        if df.is_empty():
            return {}

        # The model's raw generated SMILES (including invalids)
        gen_smiles = df.get_column("original_smiles").to_list()

        # 1. Calculate Intrinsic & Extrinsic Properties using MolEval
        # This returns a dictionary with keys like 'Validity', 'Uniqueness', 'Novelty', 'FCD', 'IntDiv', 'QED', 'SA', etc.
        moleval_results = self.metrics_engine.calculate(
            gen=gen_smiles,
            calc_valid=True,
            calc_unique=True,
            properties=True, # Triggers QED, SA, logP, MW calculations
            return_stats=False
        )

        # 2. Calculate FCGMB 3D/Docking Metrics
        docking_results = self._calculate_docking_metrics(df)

        # 3. Combine and return
        return {**moleval_results, **docking_results}

    def _calculate_docking_metrics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate docking-specific success metrics."""
        total = len(df)
        valid_pose_df = df.filter(pl.col("valid_pose_found") == True)

        metrics = {
            "Total_Generated": total,
            "Fragment_Match_Rate": len(df.filter(pl.col("skip_reason") != "failed_2d_match")) / total if total else 0.0,
            "Valid_Pose_Hit_Rate": len(valid_pose_df) / total if total else 0.0,
        }

        if not valid_pose_df.is_empty():
            # Sort by normalized score descending (higher is better)
            top_df = valid_pose_df.sort("normalized_score", descending=True)
            top_10 = top_df.head(10)
            top_100 = top_df.head(100)

            metrics["Avg_Top10_NormScore"] = top_10["normalized_score"].mean()
            metrics["Avg_Top10_DockingScore"] = top_10["docking_score"].mean()
            metrics["Avg_Top100_NormScore"] = top_100["normalized_score"].mean()
            metrics["Avg_Top100_DockingScore"] = top_100["docking_score"].mean()

        return metrics
```

## 3. Reasoning

* **Separation of Concerns**: The `FCGMBOracle` remains purely responsible for batch scoring and budget tracking. The `FCGMBEvaluator` handles post-hoc analysis. This prevents the core oracle from becoming bloated with heavy ML dependencies.
* **Standardized Benchmarking**: By using `moleval`, `fcgmb` adopts the exact same implementations of FCD, Novelty, and Diversity as MOSES and GuacaMol. This makes the generative performance directly comparable to literature baselines.
* **Unified Metrics Dictionary**: Outputting a single dictionary combining both structural (RMSD, docking scores) and 2D (QED, Novelty) metrics provides a complete picture of whether a model is generating molecules that are both chemically sound *and* strongly binding.

## 4. Pitfalls to Look Out For

1.  **Handling Invalid SMILES Correctly**:
    `FCGMBOracle` standardizes SMILES during the `score()` call. If you pass the *standardized* or *filtered* SMILES to `moleval`, your Validity and Uniqueness will artificially appear as 1.0. You **must** pass `df["original_smiles"]` (the raw model output) to `moleval.metrics.GetMetrics.calculate()` so it can accurately calculate the model's true Validity.
2.  **FCD Overhead & Dependencies**:
    The Frechet ChemNet Distance (FCD) calculation requires PyTorch and pre-trained neural network weights. It can be slow to initialize and might fail on headless nodes without internet access (when attempting to download the weights). You may want to provide a flag `run_fcd=False` for quick evaluations.
3.  **Reference Dataset Size**:
    Metrics like FCD and Novelty depend heavily on the size and chemical diversity of the reference sets (`train` and `test`). For some benchmarks, the ChEMBL bioactivity dataset might be small (e.g., < 1,000 compounds). A small reference set can result in noisy FCD values or highly skewed Novelty scores.
4.  **Multiprocessing Collisions**:
    Both `FCGMBOracle` (via RDKit/AutoDock) and `moleval` utilize multiprocessing (`n_jobs`). Running them concurrently or failing to close the `GetMetrics` multiprocessing pool can lead to zombie processes or memory leaks. Ensure that `moleval`'s pool is properly closed after `evaluate()` is called (if not handled automatically by garbage collection).
5.  **Target-Specific vs Global Novelty**:
    `moleval` checks Novelty against the `train` set. In this context, `train` is just the *known lowest-quartile binders for that specific target*. A generated molecule might be "Novel" relative to this specific target's known binders, but could still be a known drug for another target. It is important to contextualize this metric as "Target-Specific Novelty".
