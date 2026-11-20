# MOCKDOCK

<p align="center">
  <img src="assets/Mock_Dock_Duck.svg" alt="MockDock banner" width="800" />
</p>

**mockdock** is a docking-based benchmarking package for chemical language models (CLMs) and generative algorithms performing fragment-constrained molecular design. Each benchmark pairs a curated PDB crystal structure with bioactivity-annotated compounds, evaluating models on their ability to decorate core fragments into high-scoring molecules while maintaining a similar binding mode. Illustration by luna eve.

**Documentation:** [https://popov-lab-unc.github.io/mockdock/](https://popov-lab-unc.github.io/mockdock/) &nbsp;|&nbsp; **License:** [MIT](./LICENSE)

## Installation

### 1. Install AutoDock-GPU

Install [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) (tested with **v1.6**) by downloading a pre-compiled binary from [GitHub Releases](https://github.com/ccsb-scripps/AutoDock-GPU/releases) or building from source.

Ensure `adgpu` is in your `PATH`, or set its location via:
```bash
export ADGPU_EXECUTABLE=/path/to/adgpu
```

### 2. Install mockdock

```bash
git clone https://github.com/Popov-Lab-UNC/mockdock.git
cd mockdock
pip install -e .
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Popov-Lab-UNC/mockdock.git
cd mockdock
uv venv && uv sync
```

## Quickstart

mockdock exposes two core classes: `MDOracle` (runs docking and writes a self-describing run directory) and `MDEvaluator` (reads that run directory to compute all benchmark metrics, per-molecule properties, and optimization trajectories).

### 1. Scoring with `MDOracle`

```python
from mockdock import MDOracle

# Initialize oracle for a benchmark target
oracle = MDOracle("CHK1", budget=1000, run_dir="./my_run")

# Seed compounds & required 2D fragment
initial_df = oracle.get_initial_compounds()
fragment_smiles = oracle.fragment_smiles

# Score candidate molecules (returns {smiles: reward_score in [0.0, 1.0]})
scores = oracle.score(["CCO", "c1ccccc1"])

# Inspect live state (automatically flushed to ./my_run/results.csv and ./my_run/status.json)
print(oracle.results_df)       # Full Polars history DataFrame
print(oracle.budget_remaining) # Remaining oracle calls

# In-process evaluation: evaluate directly from the session
metrics = oracle.evaluate()    # Writes ./my_run/eval_metrics.json
print(f"Top 10 Mean Reward: {metrics['avg_top_10']:.3f}")
print(f"Fragment Adherence: {metrics['fragment_adherence']:.1%}")
```

### 2. Post-hoc Evaluating with `MDEvaluator`

Point `MDEvaluator` directly at any `MDOracle` run directory—benchmark configuration, fragment SMILES, and evaluation budget are loaded automatically from `status.json`:

```python
from mockdock import MDEvaluator

# Post-hoc evaluation from a run directory
evaluator = MDEvaluator("./my_run")
metrics = evaluator.compute_metrics()  # Writes ./my_run/eval_metrics.json

print(evaluator.summary(metrics))
print(metrics["effective_yield_rate"])
print(metrics["avg_top_10"])
print(metrics["auc_top_10"])

# Per-molecule quality table (QED, SA, novelty, fragment adherence, MedChem)
mol_df = evaluator.compute_molecule_metrics()

# Running top-10 optimization curve over cumulative oracle calls
traj_df = evaluator.compute_trajectory(k=10)
```

You can also evaluate at a smaller budget cutoff (e.g., comparing 500 vs. 1,000 calls on the same run) or run from the CLI:

```python
metrics_500 = evaluator.compute_metrics(budget=500)  # automatically avoids overwriting eval_metrics.json on sub-budget
```

```bash
python -m mockdock.evaluator ./my_run
python -m mockdock.evaluator ./my_run --budget 500 --output eval_500.json
```

> **Evaluating external CSVs:** If you have a standalone `results.csv` without a `status.json`, use `MDEvaluator.from_csv("results.csv", benchmark="CHK1", budget=1000)`.

## Evaluation Metrics

`MDEvaluator.compute_metrics()` reports the following metrics:

| Category | Metric | Description |
| --- | --- | --- |
| **Generation** | `validity` | Fraction of generated SMILES that parse into valid RDKit molecules. |
|  | `uniqueness` | Fraction of valid molecules that are structurally unique. |
|  | `novelty` | Fraction of unique valid molecules not present in the seed set. |
|  | `fragment_adherence` | Fraction of unique valid molecules containing the required 2D fragment. |
|  | `effective_yield_rate` | Fraction of all generated molecules that are valid, unique, novel, and fragment-adherent. |
| **Diversity** | `internal_diversity` | Mean pairwise Tanimoto distance (Morgan FP) among unique valid molecules. |
|  | `scaffold_diversity` | Fraction of unique Murcko scaffolds among unique valid molecules. |
|  | `snn` | Mean nearest-neighbor Tanimoto similarity to the seed set. |
| **Chemical Quality** | `mean_qed` | Mean quantitative estimate of drug-likeness (QED) of unique valid molecules. |
|  | `mean_sa` | Mean synthetic accessibility (SA) score of unique valid molecules (1 = easy, 10 = hard; null if sascorer unavailable). |
|  | `fraction_medchem_pass` | Fraction of unique valid fragment-containing molecules passing PAINS/BMS alerts and physchem bounds. |
|  | `effective_medchem_pass` | Fraction of effective-yield molecules passing PAINS/BMS alerts and physchem bounds. |
| **Optimization** | `avg_top_1`, `avg_top_10`, `avg_top_100` | Mean reward score of the top-$k$ unique scored molecules (zero-padded to $k$; `null` if budget $< k$). |
|  | `auc_top_10` | Area under the running top-10 reward curve over cumulative oracle calls (normalized over budget; carries final score forward for early-stopped runs). |
|  | `valid_pose_rate` | Fraction of docked molecules with a pose within the fragment RMSD threshold. |

## Available Benchmarks

| Benchmark | Target ID | Structure ID | Reference Ligand | Calibration Bounds |
| --- | --- | --- | --- | --- |
| **CHK1** | CHEMBL4630 | 2R0U | M54 | [-6.44, -11.79] |
| **DPP4** | CHEMBL284 | 2HHA | 3TP | [-6.21, -11.23] |
| **ITK** | CHEMBL2959 | 3QGW | L7A | [-6.55, -11.45] |
| **PEPCK** | CHEMBL2911 | 2GMV | UN8 | [-6.12, -10.98] |
| **PptT** | CHEMBL5465373 | 8GKF | D16 | [-6.30, -11.50] |
| **TTK** | CHEMBL3983 | 3WZJ | O43 | [-6.48, -11.62] |
| **VEGFR2** | CHEMBL279 | 3VHE | 42Q | [-6.70, -12.10] |

*See the [documentation](https://popov-lab-unc.github.io/mockdock/) for scoring mechanics, adding custom targets, example scripts, and full API reference.*
