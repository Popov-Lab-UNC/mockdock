# MOCKDOCK

<p align="center">
  <img src="assets/Mock_Dock_Duck.svg" alt="MockDock banner" width="900" />
</p>

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
# ['CHK1', 'DPP4', 'ITK', 'PEPCK', 'PptT', 'TTK', 'VEGFR2']

# Instantiate for a specific benchmark
oracle = MDOracle("CHK1", budget=1000)

# 1. Get initial compounds (lowest-quartile ChEMBL bioactivity) 
# Provide these to your generative model as starting points.
initial_df = oracle.get_initial_compounds()

# 2. Fragment that molecules must contain (mol.HasSubstruct)
fragment_smiles = oracle.fragment_smiles

# 2. Score a list of SMILES
# Returns {smiles: reward_score}; results_df also records docking_score and norm_score.
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

print(metrics["avg_top_10"])  # Mean reward score of top 10 compounds
print(metrics["novelty"])     # Fraction of compounds not in the initial set
```

## Available Benchmarks

| Name   | PDB  | Target      | Reference Ligand |
|--------|------|-------------|------------------|
| CHK1   | 2R0U | CHEMBL4630  | M54              |
| DPP4   | 2HHA | CHEMBL284   | 3TP              |
| ITK    | 3QGW | CHEMBL2959  | L7A              |
| PEPCK  | 2GMV | CHEMBL2911  | UN8              |
| PptT   | 8GKF | Custom      | D16              |
| TTK    | 3WZJ | CHEMBL3983  | O43              |
| VEGFR2 | 3VHE | CHEMBL279   | 42Q              |

## Adding a Custom Benchmark

The `PptT` setup is a reference workflow for creating your own benchmark.

1. **Create a benchmark TOML** in `src/mockdock/configs/<Name>.toml`.
   - Required core fields: `benchmark_name`, `pdb_id`, `fragment_smiles`, and benchmark constraints.
   - For PromptSMILES-based models, set `fragment_smiles_with_dummies`.
   - For LibInvent, prefer a dedicated two-attachment scaffold via `libinvent_scaffold_with_dummies`.
2. **Add curated bioactivity data** at `src/mockdock/bioactivity_data/<Name>.csv`.
   - Required columns: `molecule_chembl_id`, `canonical_smiles`, `pchembl_value`.
3. **Provide docking grids** in `src/mockdock/grids/<PDB_ID>/` (or rely on auto-preparation).
   - Include the `.maps.fld` grid files and reference ligand (`<PDB_ID>_ligand_corrected.sdf` when available).
4. **Run 5x variance to calibrate normalization**:
   ```bash
   python scripts/variance/run_variance.py \
     --config src/mockdock/configs/PptT.toml \
     --run-dir variance_runs/PptT \
     --output-dir variance_analysis/PptT \
     --n-iters 5
   ```
5. **Inspect variance outputs**:
   - Review docking/activity plots for Pearson, Spearman, and R2 trends.
   - Confirm RMSD pass behavior from the valid-pose overlays.
6. **Set `low_score` and `high_score`** in the TOML once variance is complete.
7. **Run model suites** with and without reward clipping:
   - `sbatch --export=ALL,CLIP_REWARD_UPPER_BOUND=false scripts/experiments/slurm_pptt_model_suite.sh`
   - `sbatch --export=ALL,CLIP_REWARD_UPPER_BOUND=true scripts/experiments/slurm_pptt_model_suite.sh`

### PptT / LibInvent Caveat

PptT is naturally a single-exit-vector scaffold for most models (`CC1=NC(c2c(N1)ccc([*])c2)=O`), while LibInvent requires two attachment points. For LibInvent runs, use the dedicated two-dummy scaffold (`O=C1N=C([*])Nc2c1cc([*])cc2`) and document this constraint when reporting results.

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
| `mean_qed_novel` | Mean drug-likeness (QED) of valid novel molecules. |
| `mean_sa_novel` | Mean synthetic accessibility score of valid novel molecules. |
| `fraction_medchem_pass` | Fraction of unique valid molecules passing structural alerts and physchem bounds. |
| `fraction_pains_free` | Fraction of unique valid molecules not flagging any PAINS substructure. |
| `fraction_bms_free` | Fraction of unique valid molecules not flagging any BMS substructure. |
| `fraction_lipinski` | Fraction of unique valid molecules passing all four Lipinski Ro5 criteria. |
| `fragment_incorporation`| Fraction of molecules containing the required 2D fragment. |
| `novelty` | Fraction of molecules not present in the initial ChEMBL set. |
| `effective_novelty` | Novel molecules that are also non-identical to their parents. |
| `avg_top_10` | Mean reward score of the top 10 molecules. |
| `avg_top_10_norm` | Mean uncapped normalized score of the top 10 molecules. |
| `avg_top_10_filtered`| Mean reward score of the top 10 molecules passing MedChem filters. |
| `auc_top_10` | Area under the running top-10 score curve (optimization speed). |
| `auc_top_10_filtered`| Area under the running top-10 score curve for MedChem-passing molecules. |
| `oracle_efficiency_80` | Calls required to reach 80% of final top-10 score. |
| `oracle_efficiency_100` | Calls required for the running top-10 reward to reach 1.0. |

## Scoring & Constraints

`MDOracle.score()` returns the bounded `reward_score` in **[0.0, 1.0]**:
1.  **2D Fragment Check**: Molecules must contain the benchmark fragment. If not, they score `0.0`.
2.  **3D RMSD Check**: Valid docking poses must overlay the crystal fragment within a threshold (default: 2.0 Å).
3.  **Normalization**: Raw docking energies are scaled between a target-specific `low_score` (0.0; worst mean docking score in the original variance runs) and `high_score` (1.0; best mean docking score in the original variance runs).

Run outputs keep three score columns:
- `docking_score`: raw docking energy, lower is better.
- `norm_score`: uncapped normalized score, useful for seeing when generated molecules exceed the original benchmark range.
- `reward_score`: `norm_score` clipped to `[0, 1]`, used by RL algorithms and primary optimization metrics.

### Relaxing Constraints
You can relax these for exploration:
```python
oracle = MDOracle("CHK1")
oracle._loader.require_fragment_match = False # Dock everything
oracle._loader.require_pose_rmsd = False      # Reward best score, ignore pose RMSD
```

## Analyzing Experiments

The `scripts/` directory contains tools for large-scale analysis:

*   **`scripts/analysis/analyze_experiments.py`**: Aggregates results across multiple models, benchmarks, and seeds. It generates:
    *   `metrics_summary.csv` and `metrics_summary_macro.csv`: Aggregated metrics.
    *   Publication-quality figures (Figure 1: Generation Metrics, Figure 2: Optimization Metrics, Figure 3: Quality Metrics, Figure 4: Trajectories).
*   **`scripts/variance/run_variance.py`**: Runs multiple seeds of a benchmark to assess stability.
*   **`scripts/docking/run_workflow.py`**: Standalone CLI to run the docking workflow for specific protein-ligand benchmarks.

Usage:
```bash
python scripts/analysis/analyze_experiments.py --exps-dir exps/ --output-dir analysis_results/
```

## Project Structure & Caching

mockdock distinguishes between persistent global assets and session-specific results:

*   **Global Cache (`scratch_dir`)**: Defaults to `~/.mockdock/`. Stores pre-built AutoGrid maps and ChEMBL data.
*   **Run Directory (`run_dir`)**: Defaults to `./run_<timestamp>/`. Contains outputs specific to the current session (results CSVs, docked poses, metrics).

```
mockdock/
├── src/mockdock/
│   ├── configs/           # TOML benchmark definitions
│   ├── bioactivity_data/  # Curated ChEMBL CSVs
│   ├── grids/             # Pre-built AutoGrid maps
│   ├── oracle.py          # Main MDOracle class
│   ├── evaluator.py       # Post-hoc MDEvaluator
│   ├── docking.py         # Backend engines (Vina/AD-GPU)
│   ├── filters.py         # MedChem filtering (PAINS, BMS, etc.)
│   └── ... 
├── scripts/
│   ├── dataset/            # ChEMBL/PDB dataset generation
│   ├── docking/            # Standalone docking workflows
│   ├── variance/           # Stability and calibration runs
│   ├── experiments/        # Model suite launchers
│   ├── scoring/            # Expensive molecule scoring jobs
│   └── analysis/           # Metrics and figures
├── tests/                 # Pytest suite
└── pyproject.toml         # Dependencies and project metadata
```