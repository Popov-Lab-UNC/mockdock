# mockdock

A docking-based benchmarking package for chemical language models (CLMs) for fragment-constrained molecular generation. 

Each benchmark is built around a protein–ligand system from the PDB and a corresponding set of bioactivity-annotated compounds from ChEMBL. Models are scored by docking compounds that contain a specified molecular fragment, using AutoDock-GPU (preferred) or Vina as the backend.

## Installation

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

## Quickstart (The Oracle)

The `MDOracle` is the main interface for scoring generated molecules. It handles ligand preparation, docking, and scoring automatically.

```python
from mockdock import MDOracle

# List all available benchmarks
MDOracle.list_benchmarks()
# ['CHK1', 'DPP4', 'ITK', 'PEPCK', 'TTK', 'VEGFR2']

# Instantiate for a specific benchmark
oracle = MDOracle("CHK1", budget=1000)

# 1. Get initial compounds (lowest-quartile ChEMBL bioactivity) 
# Provide these to your generative model as starting points.
initial_df = oracle.get_initial_compounds()

# 2. Fragment that molecules must contain (mol.HasSubstruct)
fragment_smiles = oracle.fragment_smiles

# 2. Score a list of SMILES
# Returns {smiles: normalized_score}
scores = oracle.score(["CCO", "CCC"])

# 3. Inspect results
print(oracle.results_df)      # Full history as Polars DataFrame
print(oracle.budget_remaining) # Remaining oracle calls
```

## Post-hoc Evaluation

Use `MDEvaluator` to compute a comprehensive suite of metrics (validity, uniqueness, diversity, novelty, and docking performance) after a benchmark run.

```python
from mockdock import MDEvaluator
from pathlib import Path

evaluator = MDEvaluator("CHK1")
metrics = evaluator.compute_metrics(Path("run_20240408_120000/results.csv"))

print(metrics["avg_top_10"])  # Mean normalized score of top 10 compounds
print(metrics["novelty"])     # Fraction of compounds not in the initial set
```

## Available Benchmarks

| Name   | PDB  | Target      | Reference Ligand |
|--------|------|-------------|------------------|
| CHK1   | 2R0U | CHEMBL4630  | M54              |
| DPP4   | 2HHA | CHEMBL284   | 3TP              |
| ITK    | 3QGW | CHEMBL2959  | L7A              |
| PEPCK  | 2GMV | CHEMBL2911  | UN8              |
| TTK    | 3WZJ | CHEMBL3983  | O43              |
| VEGFR2 | 3VHE | CHEMBL279   | 42Q              |

## API Reference

### `MDOracle(benchmark_name, budget, docking_backend, scratch_dir, run_dir, n_cpus, n_gpus)`

| Parameter | Default | Description |
|---|---|---|
| `benchmark_name` | required | Name of the benchmark (e.g., "CHK1") |
| `budget` | `1000` | Maximum number of compounds allowed to be scored |
| `docking_backend` | `"auto"` | `"autodock_gpu"`, `"vina"`, or `"auto"` |
| `scratch_dir` | `~/.mockdock` | Global persistent cache (receptor grids, ChEMBL data) |
| `run_dir` | `./run_<ts>` | Output directory for this specific session |
| `n_cpus` | autodetect | CPUs for parallel preparation |
| `n_gpus` | autodetect | GPUs for AutoDock-GPU |

### `MDEvaluator(benchmark_name, scratch_dir)`

Computes standardized metrics for a finished run. It is lightweight and does not require a docking engine.

| Metric | Description |
|---|---|
| `validity` | Fraction of generated SMILES that parse into valid RDKit molecules. |
| `uniqueness` | Fraction of valid molecules that are structurally distinct. |
| `internal_diversity` | Average pairwise Tanimoto distance among unique valid molecules. |
| `scaffold_diversity` | Fraction of unique Murcko scaffolds among unique valid molecules. |
| `mean_qed` | Mean drug-likeness (QED) of valid molecules. |
| `fragment_incorporation`| Fraction of molecules containing the required 2D fragment. |
| `novelty` | Fraction of molecules not present in the initial ChEMBL set. |
| `effective_novelty` | Novel molecules that are also non-identical to their parents. |
| `avg_top_10` | Mean normalized docking score of the top 10 molecules. |
| `auc_top_10` | Area under the running top-10 score curve (optimization speed). |
| `oracle_efficiency_80` | Calls required to reach 80% of final top-10 score. |

## Scoring & Constraints

`MDOracle.score()` returns a normalized value in **[0.0, 1.0+]**:
1.  **2D Fragment Check**: Molecules must contain the benchmark fragment. If not, they score `0.0`.
2.  **3D RMSD Check**: Valid docking poses must overlay the crystal fragment within a threshold (default: 2.0 Å).
3.  **Normalization**: Raw docking energies are scaled between a target-specific `low_score` (0.0) and `high_score` (1.0) derived from ChEMBL bioactivity.

### Relaxing Constraints
You can relax these for exploration:
```python
oracle = MDOracle("CHK1")
oracle._loader.require_fragment_match = False # Dock everything
oracle._loader.require_pose_rmsd = False      # Reward best score, ignore pose RMSD
```

## Analyzing Experiments

The `scripts/` directory contains tools for large-scale analysis:

*   **`analyze_experiments.py`**: Aggregates results across multiple models, benchmarks, and seeds. It generates:
    *   `metrics_summary.csv`: Aggregated metrics for every model/target.
    *   Publication-quality figures (Performance panels, Optimization trajectories).
*   **`run_variance.py`**: Runs multiple seeds of a benchmark to assess stability.

Usage:
```bash
python scripts/analyze_experiments.py --exps-dir exps/ --output-dir analysis_results/
```

## Project Structure

```
mockdock/
├── src/mockdock/
│   ├── configs/           # YAML benchmark definitions
│   ├── bioactivity_data/  # Curated ChEMBL CSVs
│   ├── grids/             # Pre-built AutoGrid maps
│   ├── oracle.py          # Main MDOracle class
│   ├── evaluator.py       # Post-hoc MDEvaluator
│   ├── docking.py         # Backend engines (Vina/AD-GPU)
│   └── ... 
├── scripts/
│   ├── analyze_experiments.py
│   ├── run_workflow.py    # Standalone CLI workflow
│   └── run_variance.py    # Stability testing
├── tests/                 # Pytest suite
└── pyproject.toml         # Dependencies and project metadata
```

## Caching & Storage

mockdock distinguishes between persistent global assets and session-specific results:

*   **Global Cache (`scratch_dir`)**: Defaults to `~/.mockdock/`. This is used to store persistent data that can be reused across multiple runs, such as pre-built AutoGrid maps and cleaned ChEMBL bioactivity CSVs.
*   **Run Directory (`run_dir`)**: Defaults to `./run_<timestamp>/` in your current working directory. This contains all outputs specific to the current session:
    *   `results.csv`: Full list of scored SMILES (including those that failed the 2D filter).
    *   `results.yaml`: Human-readable summary of top-scoring molecules.
    *   `poses/`: Directory containing docked pose files.
    *   `metrics.json`: Per-run timing, budget usage, and efficiency statistics.
    *   `<benchmark>_top_10_poses.sdf`: SDF file containing the top 10 poses.

## Development

```bash
uv sync --dev
uv run pytest                          # Run tests
uv run ruff format .                   # Format code
```