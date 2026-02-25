# FCGMB: Fragment-Constrained Generative Model Benchmark

A benchmarking package for generative molecular models. Each benchmark is built around a protein–ligand system from the PDB and a corresponding set of bioactivity-annotated compounds from ChEMBL. Models are scored by docking compounds that contain a specified molecular fragment, using AutoDock-GPU as the backend.

## Installation

```bash
git clone https://github.com/Popov-Lab-UNC/fcgmb.git
cd fcgmb
pip install -e .
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Popov-Lab-UNC/fcgmb.git
cd fcgmb
uv venv && uv sync
```

## Quickstart

```python
from fcgmb import FCGMBOracle

# List all available benchmarks
FCGMBOracle.list_benchmarks()
# ['AKT1', 'CHK1', 'ITK', 'PCK1', 'TTK', 'VEGFR2']

# Instantiate for a specific benchmark
oracle = FCGMBOracle("CHK1", budget=5000)

# The fragment SMILES that every submitted molecule must contain
print(oracle.fragment)

# Lower-quartile compounds from ChEMBL — provide these to your generative model
initial_df = oracle.get_initial_compounds()

# Score a list of SMILES; returns {smiles: normalized_score}
scores = oracle.score(["CCO", "CCC"])

# Inspect results so far
print(oracle.results_df)
print(oracle.budget_remaining)  # how many compounds remain in the budget
print(oracle.status)            # 'active' or 'finished'
```

## Available Benchmarks

| Name   | PDB  | Target      |
|--------|------|-------------|
| AKT1   | 4EJN | CHEMBL4282  |
| CHK1   | 2R0U | CHEMBL4630  |
| ITK    | 3QGW | CHEMBL2959  |
| PCK1   | 1NHX | CHEMBL2911  |
| TTK    | 3WZJ | CHEMBL3983  |
| VEGFR2 | 3VHE | CHEMBL279   |

## API Reference

### `FCGMBOracle(benchmark_name, budget, docking_backend, scratch_dir, n_cpus, n_gpus)`

| Parameter | Default | Description |
|---|---|---|
| `benchmark_name` | required | Name of the benchmark (see table above) |
| `budget` | `5000` | Maximum number of compounds that can be scored |
| `docking_backend` | `"auto"` | `"autodock_gpu"`, `"vina"`, or `"auto"` |
| `scratch_dir` | `.fcgmb/` | Directory for cached grids, bioactivity data, and results |
| `n_cpus` | autodetect | CPUs for ligand preparation |
| `n_gpus` | autodetect | GPUs for AutoDock-GPU |

### Public attributes

| Attribute | Description |
|---|---|
| `oracle.benchmark_name` | Name of this benchmark |
| `oracle.pdb_id` | PDB ID of the receptor structure |
| `oracle.fragment` | Fragment SMILES molecules must contain |
| `oracle.config` | `{rmsd_threshold, require_fragment_match, require_pose_rmsd, low_score, high_score}` |
| `oracle.n_cpus` / `n_gpus` | Hardware in use |
| `oracle.max_budget` | Total scoring budget |
| `oracle.budget_used` | Compounds scored so far |
| `oracle.budget_remaining` | Remaining budget |
| `oracle.status` | `"active"` or `"finished"` |
| `oracle.results_df` | Polars DataFrame of all scored compounds |

### Methods

```python
FCGMBOracle.list_benchmarks()          # class method — list bundled benchmarks
oracle.get_initial_compounds()         # lower-quartile bioactivity compounds (DataFrame)
oracle.get_validation_compounds()      # upper-quartile bioactivity compounds (DataFrame)
oracle.score(smiles_list)              # dock and score; returns {smiles: float}
oracle.set_backend_config(**kwargs)    # override vina_exhaustiveness, n_poses, etc.
```

### Scoring

`score()` returns a normalized score in **[0.0, 1.0]**:
- Valid docking poses (RMSD ≤ threshold relative to the crystal ligand) are normalized between the empirical `low_score` and `high_score` from the config. Otherwise, they score `0.0`.
- Molecules can exceed `1.0` if they have better docking scores than the ChEMBL data.
- Molecules that do **not** contain the fragment substructure are skipped and score `0.0` (no oracle calls consumed) — unless `require_fragment_match` is `False`.
- Once `budget` compounds have been scored, all further calls return `0.0`.

### Relaxing Scoring Constraints

Both constraints are `True` by default (standard benchmark behaviour). They can be relaxed to explore molecules outside the fragment-constrained design space.

**Disable the 2D fragment filter** — any molecule is docked, regardless of whether it contains the fragment. This also disables the 3D RMSD check (since there is no fragment to measure RMSD against):

```python
from fcgmb import FCGMBOracle

oracle = FCGMBOracle("AKT1", budget=5000)
oracle._require_fragment_match = False  # molecules without fragment are now docked
oracle._require_pose_rmsd = False       # automatically implied, but set explicitly for clarity

scores = oracle.score(["CCO", "CCC"])  # both are docked even without the AKT1 fragment
```

**Disable only the 3D RMSD filter** — the fragment must still be present (2D check passes), but any docking pose is accepted rather than requiring the fragment to overlay the crystal pose within the RMSD threshold:

```python
from fcgmb import FCGMBOracle

oracle = FCGMBOracle("CHK1", budget=5000)
oracle._require_pose_rmsd = False  # best docking score accepted regardless of pose RMSD

scores = oracle.score(my_smiles_list)
```

> **Note:** These flags are also configurable per-benchmark via the YAML configs (`require_fragment_match`, `require_pose_rmsd`). Overriding the instance attributes (as shown above) takes effect immediately for that oracle session.

## Local Storage

FCGMB caches all runtime data under a `.fcgmb/` scratch directory (configurable via `scratch_dir`):

```
.fcgmb/
├── grids/<pdb_id>/         # AutoGrid maps (prepared once, reused)
├── bioactivity_data/       # Cached ChEMBL CSVs
└── runs/<benchmark>/
    └── results/            # Docking output files
```

Bundled assets (pre-built grids and curated bioactivity CSVs) are shipped inside the package under `fcgmb/configs/`, `fcgmb/grids/`, and `fcgmb/bioactivity_data/`, and are used automatically.

## Package Layout

```
fcgmb/
├── configs/           # Bundled YAML benchmark configs
├── bioactivity_data/  # Curated ChEMBL CSVs (ground truth)
├── grids/             # Pre-built AutoGrid maps
├── oracle.py          # FCGMBOracle — main user-facing class
├── docking.py         # AutoDock-GPU and Vina backends
├── receptor.py        # Receptor preparation pipeline
├── ligand_prep.py     # Ligand preparation (PDBQT conversion)
├── analysis.py        # RMSD filtering and pose analysis
└── data.py            # ChEMBL data fetching
```