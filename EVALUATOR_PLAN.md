# FCGMB Evaluator & Analysis Pipeline — Implementation Plan

## Overview

This plan describes three components that together form a complete post-hoc evaluation and visualization pipeline for the FCGMB benchmark experiments stored in `exps/`.

1. **`src/fcgmb/loader.py`** — A shared `BenchmarkLoader` class used by **both** `FCGMBOracle` and `FCGMBEvaluator` to load benchmark config and bioactivity data; eliminates duplication and ensures consistency.
2. **`src/fcgmb/evaluator.py`** — Houses the `FCGMBEvaluator` class. Reads a single `results.csv` from one run/target, computes a comprehensive set of intrinsic and extrinsic metrics, and writes `eval_metrics.json`.
3. **`scripts/analyze_experiments.py`** — Discovers all experiment runs under `exps/`, invokes `FCGMBEvaluator` on each, aggregates results across the 5 seeds per model, computes mean ± std, and produces publication-quality figures and a master CSV.

---

## 1. Shared Loader (`src/fcgmb/loader.py`)

### 1.1 Motivation

Both `FCGMBOracle` (docking) and `FCGMBEvaluator` (post-hoc metrics) need two things from the benchmark config:

- The benchmark **YAML config** (fragment SMILES, score bounds, PDB ID, etc.)
- The **bioactivity data** and the 25th-percentile threshold that defines the initial compound set

Centralizing this in `BenchmarkLoader` eliminates duplication and ensures both classes always agree on which molecules are "initial compounds".

### 1.2 Interface Sketch

```python
# src/fcgmb/loader.py
"""
Shared loader for FCGMB benchmark config and bioactivity data.
Used by both FCGMBOracle and FCGMBEvaluator.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import yaml
from .data import fetch_chembl_data


class BenchmarkLoader:
    """
    Lightweight loader: reads the benchmark YAML config and bioactivity CSV.
    Does NOT initialise any docking engine or create run directories.
    """

    def __init__(self, benchmark_name: str, scratch_dir: Optional[Path] = None):
        self.benchmark_name = benchmark_name
        _pkg = Path(__file__).parent
        self._pkg_bioactivity_dir = _pkg / "bioactivity_data"
        _scratch = Path(scratch_dir).resolve() if scratch_dir else Path.home() / ".fcgmb"
        self._bioactivity_data_dir = _scratch / "bioactivity_data"

        config_path = self._find_config(benchmark_name, _pkg / "configs")
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self.pdb_id: str = raw["pdb_id"]
        self.target_id: str = raw.get("target_id", "")
        self.doc_id: Optional[str] = raw.get("doc_id")
        self.fragment_smiles: str = raw["fragment_smiles"]
        self.fragment_smiles_with_dummies: Optional[str] = raw.get("fragment_smiles_with_dummies")
        self.rmsd_threshold: float = raw.get("rmsd_threshold", 2.0)
        self.low_score: Optional[float] = raw.get("low_score")
        self.high_score: Optional[float] = raw.get("high_score")
        self.ligand_resname: Optional[str] = raw.get("ligand_resname")
        self._chembl_data: Optional[pl.DataFrame] = None

    @staticmethod
    def _find_config(name: str, config_dir: Path) -> Path:
        for stem in [name, name.upper(), name.lower()]:
            p = config_dir / f"{stem}.yaml"
            if p.exists():
                return p
        available = [f.stem for f in config_dir.glob("*.yaml")]
        raise FileNotFoundError(f"Benchmark config '{name}' not found. Available: {available}")

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        return sorted(f.stem for f in (Path(__file__).parent / "configs").glob("*.yaml"))

    def get_full_data_and_threshold(self) -> tuple[pl.DataFrame, float, str]:
        """Lookup order: in-memory cache → bundled CSV → scratch cache → live ChEMBL fetch."""
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
                df.write_csv(self._bioactivity_data_dir / f"{self.benchmark_name}_chembl.csv")
        if df.is_empty():
            return df, 0.0, ""
        pvals = df.get_column("pchembl_value").to_numpy()
        threshold = float(np.quantile(pvals, 0.25)) if pvals.size > 0 else 0.0
        self._chembl_data = df
        self._threshold = threshold
        return df, threshold, "pchembl_value"

    def get_initial_compounds(self) -> pl.DataFrame:
        """Return the model-visible initial compound set (lowest-quartile bioactivity)."""
        df, threshold, act_col = self.get_full_data_and_threshold()
        return df.filter(pl.col(act_col) <= threshold) if not df.is_empty() else df

    def get_validation_compounds(self) -> pl.DataFrame:
        """Return the validation compound set (above the 25th-percentile threshold)."""
        df, threshold, act_col = self.get_full_data_and_threshold()
        return df.filter(pl.col(act_col) > threshold) if not df.is_empty() else df
```

> **Downstream effect on `FCGMBOracle`**: The private `_get_full_data_and_threshold()`, `get_initial_compounds()`, and `get_validation_compounds()` methods in `oracle.py` are replaced by delegation to a `BenchmarkLoader` instance (`self._loader`). Config attributes (`_fragment_smiles`, `_pdb_id`, `_low_score`, etc.) are read from `self._loader` rather than re-parsed from YAML. This is a refactor with identical external behaviour.

---

## 2. Evaluator (`src/fcgmb/evaluator.py`)

### 2.1 Input / Output

| Item | Description |
|---|---|
| **Input** | A single `results.csv` (columns: `smiles`, `original_smiles`, `docking_score`, `normalized_score`, `valid_pose_found`, `skip_reason`, `n_conformers`, `generation_round`, ...) and the benchmark name (e.g. `CHK1`). |
| **Reference set** | The **initial compounds** (via `BenchmarkLoader.get_initial_compounds()` / lower-quartile bioactivity) — molecules *visible* to the generative model. Used as the reference for Novelty and SNN. |
| **Output** | `eval_metrics.json` written alongside the `results.csv`, containing all metric values plus a `"descriptions"` sub-dictionary with one-sentence explanations. |

### 2.2 Metrics Catalogue

#### Intrinsic Metrics (computed from generated set alone)

| Metric | Description |
|---|---|
| **Validity** | Fraction of generated SMILES that parse into valid RDKit molecules. |
| **Uniqueness** | Fraction of valid molecules that are structurally distinct (unique canonical SMILES). |
| **Internal Diversity (IntDiv1)** | Average pairwise Tanimoto distance among unique valid molecules. |
| **Scaffold Diversity** | Fraction of unique Murcko scaffolds among unique valid molecules. |
| **Mean QED** | Mean quantitative estimate of drug-likeness of unique valid molecules. |
| **Mean SA** | Mean synthetic accessibility score of unique valid molecules (1=easy, 10=hard). |
| **Fragment Incorporation Rate** | Fraction of unique valid molecules containing the required fragment substructure. |
| **Fraction Passing Lipinski Ro5** | Fraction of unique valid molecules satisfying all four Lipinski Rule-of-Five criteria (MW ≤ 500, HBD ≤ 5, HBA ≤ 10, LogP ≤ 5). |
| **Fraction PAINS-Free** | Fraction of unique valid molecules that do *not* flag any PAINS substructure. |

#### Extrinsic Metrics (computed against the initial compounds visible to the model)

| Metric | Description |
|---|---|
| **Novelty** | Fraction of unique valid molecules whose canonical SMILES do not appear in the initial compound set. |
| **SNN (Nearest-Neighbour Similarity)** | Average maximum Tanimoto similarity between each generated molecule and the initial compound set. |

> **FCD (Fréchet ChemNet Distance) — deprecated.** FCD requires a separate PyTorch model (`fcd_torch`) and is sensitive to set size, making it unreliable at ≤1000-molecule budgets. It has been removed from the default metric set. It may be re-enabled as a standalone utility script if needed in future.

#### Docking / Oracle Performance Metrics

| Metric | Description |
|---|---|
| **Avg-Top-1** | Normalized docking score of the single best-scoring molecule. |
| **Avg-Top-10** | Mean normalized docking score of the 10 best-scoring molecules. |
| **Avg-Top-100** | Mean normalized docking score of the 100 best-scoring molecules (if available). |
| **AUC-Top-10** | Area under the running top-10 average normalized score curve as a function of **cumulative oracle calls** (row order). |
| **Valid Pose Hit Rate** | Fraction of docked molecules that produced at least one pose passing the RMSD threshold. |
| **Oracle Efficiency** | Oracle calls required to reach 80% of final top-10 score; fewer is better. |

### 2.3 Implementation Sketch

```python
# src/fcgmb/evaluator.py
"""
FCGMBEvaluator — post-hoc metric computation for a single FCGMB benchmark run.

Usage (CLI):
    python -m fcgmb.evaluator results.csv --benchmark CHK1 [--output eval_metrics.json]

Usage (Python API):
    from fcgmb import FCGMBEvaluator
    ev = FCGMBEvaluator("CHK1")
    metrics = ev.compute_metrics(Path("results.csv"))
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

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
    "snn":                      "Avg max Tanimoto similarity between generated and initial compounds.",
    "avg_top_1":                "Normalized docking score of the single best-scoring molecule.",
    "avg_top_10":               "Mean normalized docking score of the 10 best-scoring molecules.",
    "avg_top_100":              "Mean normalized docking score of the 100 best-scoring molecules.",
    "auc_top_10":               "AUC of running top-10 avg score curve over cumulative oracle calls.",
    "valid_pose_hit_rate":      "Fraction of docked molecules with a pose within the RMSD threshold.",
    "oracle_efficiency_80":     "Oracle calls to reach 80% of final top-10 score (fewer is better).",
}


class FCGMBEvaluator:
    """
    Post-hoc evaluator for a single FCGMB benchmark run.

    Delegates all config/bioactivity loading to BenchmarkLoader — no docking engine
    is instantiated, making this lightweight and safe to run on a login node.
    """

    def __init__(self, benchmark_name: str, scratch_dir: Optional[Path] = None):
        self.benchmark_name = benchmark_name
        self._loader = BenchmarkLoader(benchmark_name, scratch_dir=scratch_dir)
        self._pains_catalog = self._build_pains_catalog()

    def compute_metrics(self, results_csv: Path, output_path: Optional[Path] = None) -> dict:
        df = pl.read_csv(results_csv)
        ref_smiles = set(self._loader.get_initial_compounds()["smiles"].to_list())
        fragment_smiles = self._loader.fragment_smiles

        all_smiles    = df["original_smiles"].to_list()
        valid_mols    = [m for s in all_smiles if (m := Chem.MolFromSmiles(s)) is not None]
        valid_smiles  = [Chem.MolToSmiles(m) for m in valid_mols]
        unique_smiles = list(set(valid_smiles))
        unique_mols   = [Chem.MolFromSmiles(s) for s in unique_smiles]

        metrics: dict = {}
        metrics["validity"]               = len(valid_mols) / max(len(all_smiles), 1)
        metrics["uniqueness"]             = len(unique_smiles) / max(len(valid_smiles), 1)
        metrics["internal_diversity"]     = self._tanimoto_diversity(unique_smiles)
        metrics["scaffold_diversity"]     = self._scaffold_diversity(unique_smiles)
        metrics["mean_qed"]               = float(np.mean([QED.qed(m) for m in unique_mols]))
        metrics["mean_sa"]                = float(np.mean([self._sa_score(m) for m in unique_mols]))
        metrics["fragment_incorporation"] = self._fragment_rate(unique_smiles, fragment_smiles)
        metrics["fraction_lipinski"]      = self._lipinski_fraction(unique_mols)
        metrics["fraction_pains_free"]    = self._pains_free_fraction(unique_mols)
        metrics["novelty"] = self._novelty(unique_smiles, ref_smiles)
        metrics["snn"]     = self._snn(unique_smiles, list(ref_smiles))

        scored_df = df.filter(pl.col("skip_reason").is_null())
        if not scored_df.is_empty():
            top = scored_df.sort("normalized_score", descending=True)
            metrics["avg_top_1"]            = float(top.head(1)["normalized_score"].mean())
            metrics["avg_top_10"]           = float(top.head(10)["normalized_score"].mean())
            metrics["avg_top_100"]          = float(top.head(100)["normalized_score"].mean())
            metrics["auc_top_10"]           = self._auc_top_k(df, k=10)
            metrics["valid_pose_hit_rate"]  = (
                len(scored_df.filter(pl.col("valid_pose_found") == True)) / len(scored_df))
            metrics["oracle_efficiency_80"] = self._oracle_efficiency(df, k=10, frac=0.80)

        metrics["descriptions"] = METRIC_DESCRIPTIONS
        out = Path(output_path) if output_path else Path(results_csv).parent / "eval_metrics.json"
        out.write_text(json.dumps(metrics, indent=2))
        return metrics

    # ── helpers (static where possible) ──────────────────────────────

    @staticmethod
    def _build_pains_catalog():
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog(params)

    @staticmethod
    def _tanimoto_diversity(smiles_list):
        if len(smiles_list) < 2:
            return 0.0
        from rdkit.DataStructs import TanimotoSimilarity
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles_list]
        n = len(fps)
        return float(np.mean([1.0 - TanimotoSimilarity(fps[i], fps[j])
                               for i in range(n) for j in range(i + 1, n)]))

    @staticmethod
    def _scaffold_diversity(smiles_list):
        if not smiles_list:
            return 0.0
        scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
                     for s in smiles_list}
        return len(scaffolds) / len(smiles_list)

    @staticmethod
    def _sa_score(mol):
        try:
            from rdkit.Contrib.SA_Score import sascorer
            return sascorer.calculateScore(mol)
        except Exception:
            return float("nan")

    @staticmethod
    def _fragment_rate(smiles_list, fragment_smiles):
        if not fragment_smiles or not smiles_list:
            return float("nan")
        frag = Chem.MolFromSmarts(fragment_smiles)
        if frag is None:
            return float("nan")
        hits = sum(1 for s in smiles_list
                   if (m := Chem.MolFromSmiles(s)) is not None and m.HasSubstructMatch(frag))
        return hits / len(smiles_list)

    @staticmethod
    def _lipinski_fraction(mols):
        if not mols:
            return float("nan")
        def passes(m):
            return (Descriptors.MolWt(m) <= 500 and Descriptors.NumHDonors(m) <= 5
                    and Descriptors.NumHAcceptors(m) <= 10 and Descriptors.MolLogP(m) <= 5)
        return sum(1 for m in mols if passes(m)) / len(mols)

    def _pains_free_fraction(self, mols):
        if not mols:
            return float("nan")
        return sum(1 for m in mols if not self._pains_catalog.HasMatch(m)) / len(mols)

    @staticmethod
    def _novelty(generated, reference):
        if not generated:
            return float("nan")
        return sum(1 for s in generated if s not in reference) / len(generated)

    @staticmethod
    def _snn(generated, reference):
        if not generated or not reference:
            return float("nan")
        from rdkit.DataStructs import BulkTanimotoSimilarity
        ref_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
                   for s in reference if Chem.MolFromSmiles(s) is not None]
        sims = []
        for s in generated:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            sims.append(max(BulkTanimotoSimilarity(fp, ref_fps), default=0.0))
        return float(np.mean(sims)) if sims else float("nan")

    @staticmethod
    def _auc_top_k(df, k=10):
        """AUC of running top-k mean. X-axis = row order (cumulative oracle calls)."""
        scores = df["normalized_score"].to_numpy()
        seen, running_top = [], []
        for s in scores:
            seen.append(s)
            running_top.append(np.mean(sorted(seen, reverse=True)[:k]))
        x = np.arange(1, len(running_top) + 1)
        return float(np.trapz(running_top, x) / (x[-1] - x[0])) if len(x) > 1 else float("nan")

    @staticmethod
    def _oracle_efficiency(df, k=10, frac=0.80):
        scores = df["normalized_score"].to_numpy()
        target = frac * float(np.mean(sorted(scores, reverse=True)[:k]))
        seen = []
        for i, s in enumerate(scores):
            seen.append(s)
            if np.mean(sorted(seen, reverse=True)[:k]) >= target:
                return i + 1
        return len(scores)
```

> **Note:** `FCGMBEvaluator` avoids instantiating `FCGMBOracle` (no docking engine, no grid prep). Invocable from CLI as `python -m fcgmb.evaluator`.

### 2.4 Reference Set: Why Initial Compounds?

`BenchmarkLoader.get_initial_compounds()` returns the lower-quartile bioactivity molecules from ChEMBL — the molecules the generative model was explicitly shown. Using these as the Novelty/SNN reference answers *"did the model go beyond what it was shown?"*, which is more meaningful than evaluating against a hidden test set.

---

## 3. Experiment Analyzer (`scripts/analyze_experiments.py`)

### 3.1 Discovery & Aggregation

```
exps/
├── acegen-a2c/
│   ├── run_*_r01/           ← seed 1
│   │   ├── CHK1/results.csv
│   │   ├── DPP4/results.csv
│   │   └── ...
│   ├── run_*_r02/           ← seed 2
│   └── ...
└── ...
```

**Steps:**

1. For each `exps/<model>/run_*/<target>/results.csv`, call `FCGMBEvaluator(target).compute_metrics(path)`.
2. Collect into `{model: {target: {seed: metrics}}}`.
3. Aggregate over 5 seeds: **mean** and **std** for every numeric metric.
4. Macro-average across **all 6 targets** (CHK1, DPP4, ITK, PEPCK, TTK, VEGFR2).

### 3.2 Outputs

| Output | Description |
|---|---|
| `output/metrics_all.csv` | One row per (model, target, seed) with all metric columns. |
| `output/metrics_summary.csv` | One row per (model, target) with `<metric>_mean` and `<metric>_std` columns. |
| `output/metrics_summary_macro.csv` | One row per model, macro-averaged across all 6 targets. |
| `output/metric_descriptions.csv` | Metric name → one-sentence description (for paper supplementary). |
| `output/figures/` | All publication-quality figures (SVG + PNG). |

### 3.3 Figures

All figures use `seaborn` + `matplotlib` with publication defaults:

```python
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
```

#### Figure 1: Bar Charts with Error Bars (one per metric)

Grouped bar chart per metric: X = models, Y = mean over 5 seeds, error bars = ±1 std, optionally faceted by target or macro-averaged.

#### Figure 2: Avg-Top-10 Optimization Trajectory (Line Plot)

The key "optimization trajectory" figure:
- X-axis = **cumulative oracle calls** (1–1000), reconstructed from **row order** in `results.csv`
  (`generation_round` is retained in the CSV as metadata but is **not** used as the x-axis)
- Y-axis = running Avg-Top-10 normalized score
- One line per model, **mean as solid line**, **90% confidence interval as shaded band**

> **Why 90% CI rather than ±1 std?**
> With n = 5 seeds, ±1 std quantifies the *spread across runs*, which is not the right quantity for
> comparing model means in a paper. A 90% CI for the mean — `mean ± t(0.95, df=4) × std / √5`
> ≈ `mean ± 2.132 × std / √5` — is statistically correct: it expresses uncertainty about
> which model has the higher *expected* performance, which is what a reader wants to know.
> It is also a standard reporting convention in ML benchmarking papers.

```python
from scipy import stats

for model in models:
    curves = np.array([compute_running_top_k(df, k=10) for df in seed_dfs])
    n = len(curves)
    mean_curve = curves.mean(axis=0)
    sem_curve  = curves.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit     = stats.t.ppf(0.95, df=n - 1)  # 90% two-tailed, df=4 → ≈2.132
    ci_half    = t_crit * sem_curve

    plt.plot(x, mean_curve, label=model)
    plt.fill_between(x, mean_curve - ci_half, mean_curve + ci_half, alpha=0.2)
```

#### Figure 3: Radar / Hexagon Power Plot

Spider chart with one polygon per model, axes: Avg-Top-10, Validity, Novelty, Internal Diversity, Fragment Incorporation, Oracle Efficiency. Values min-max normalized to [0, 1]; lower-is-better metrics (SA, oracle_efficiency) are inverted.

#### Figure 4: Heatmap (Model × Metric)

`seaborn.heatmap`: rows = models, columns = key metrics, color normalized per column, annotated with mean values.

#### Figure 5: Per-Target Breakdown (Small Multiples)

Faceted grid across **all 6 targets** (CHK1, DPP4, ITK, PEPCK, TTK, VEGFR2) showing `avg_top_10` bars per model.

#### Figure 6: Score Distribution Violin Plots

Violin plots of `normalized_score` per model across all seeds.

#### Figure 7: Cumulative Discovery Plot

- X = cumulative oracle calls (row order, 1–1000)
- Y = molecules found with `normalized_score ≥ 0.5` (configurable threshold)
- One line per model, mean ± 90% CI shaded band

#### Figure 8: Lipinski & PAINS Summary

Grouped bar chart of `fraction_lipinski` and `fraction_pains_free` across models (macro-averaged).

---

## 4. File Layout

```
src/fcgmb/
├── __init__.py               [MODIFY] export FCGMBEvaluator, BenchmarkLoader
├── loader.py                 [NEW]    BenchmarkLoader (shared config + bioactivity)
├── evaluator.py              [NEW]    FCGMBEvaluator class
├── oracle.py                 [MODIFY] delegate to BenchmarkLoader
└── ... (existing files unchanged)

scripts/
├── analyze_experiments.py    [NEW]    Multi-run aggregation + figures
├── run_variance.py           (existing, unchanged)
└── run_workflow.py           (existing, unchanged)
```

### 4.1 Changes to `__init__.py`

```python
from .evaluator import FCGMBEvaluator
from .loader import BenchmarkLoader
# Add both to __all__
```

### 4.2 Changes to `oracle.py`

```python
# In __init__:
self._loader = BenchmarkLoader(benchmark_name, scratch_dir=scratch_dir)

# Public API delegates (replace existing methods):
def get_initial_compounds(self) -> pl.DataFrame:
    return self._loader.get_initial_compounds()

def get_validation_compounds(self) -> pl.DataFrame:
    return self._loader.get_validation_compounds()
```

All config attributes read from `self._loader` (e.g. `self._loader.fragment_smiles`, `self._loader.pdb_id`).

---

## 5. Dependencies

All required packages are already in the project environment:
- `polars`, `numpy`, `matplotlib`, `seaborn` — computation and plotting
- `rdkit` — molecule parsing, fingerprints, QED, scaffolds, PAINS filter, Lipinski descriptors
- `scipy` — `scipy.stats.t.ppf` for 90% CI; `np.trapz` for AUC

**Removed optional dependency:**
- ~~`fcd_torch`~~ — FCD is deprecated; no longer needed.

---

## 6. Design Decisions

1. **Reference set**: `initial_compounds` (model-visible, lower-quartile bioactivity). Novelty against what the model *saw* is more informative than a hidden test set.

2. **X-axis**: **Row order** = cumulative oracle call index. `generation_round` retained in raw CSV for provenance, not used as a plot axis (models with different batch sizes become directly comparable).

3. **All 6 targets**: CHK1, DPP4, ITK, PEPCK, TTK, VEGFR2 are all included. No targets excluded.

4. **FCD deprecated**: Removed due to heavy PyTorch dependency and instability at ≤1000-molecule budgets. Novelty + SNN remain as extrinsic metrics.

5. **90% CI over ±1 std in Figure 2**: Statistically correct for the sample mean over 5 seeds. `t(0.95, df=4)` ≈ 2.132. Expresses uncertainty about the *expected* model performance rather than spread of individual runs.

6. **Lipinski & PAINS**: Explicit fractions (`fraction_lipinski`, `fraction_pains_free`) complement QED/SA with direct drug-likeness and chemical quality readouts.

7. **Modularity**: `BenchmarkLoader` is the single source of truth for config and bioactivity data, shared by both `FCGMBOracle` and `FCGMBEvaluator`.

8. **Figure format**: Both SVG (vector editing) and PNG @ 300 dpi (manuscript insertion).

---

## 7. Verification Plan

### Automated
```bash
# Evaluate a single run
python -m fcgmb.evaluator \
    exps/acegen-reinvent/run_20260330_155654_r01/CHK1/results.csv \
    --benchmark CHK1

# Full analysis pipeline
python scripts/analyze_experiments.py --exps-dir exps/ --output-dir output/

# Check outputs exist
ls output/metrics_all.csv output/metrics_summary.csv output/figures/*.svg
```

### Manual
- Inspect figures for visual quality and correctness
- Verify metrics against manual spot-checks (e.g. count valid SMILES in a small run)
- Confirm radar plots are normalized correctly and readable
- Verify 90% CI bands in Figure 2 are narrower than ±1 std bands would be
