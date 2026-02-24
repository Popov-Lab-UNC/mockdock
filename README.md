# FCGMB: Fragment-Constrained Generative Model Benchmark

A benchmark package for generative molecular models, specifically using fragment-constrained docking with publicly available data and Autodock as a benchmark.

## Installation

```bash
git clone https://github.com/Popov-Lab-UNC/fcgmb.git
cd fcgmb
pip install -e .
```

or using UV:
```bash
git clone https://github.com/Popov-Lab-UNC/fcgmb.git
cd fcgmb
uv venv
uv sync
```

## Core Features

### 1. FCGMB Oracle
The `FCGMBOracle` class provides a simple interface for generative models to score compounds. It handles ChEMBL data retrieval, receptor preparation, and fragment-constrained docking automatically.

```python
from fcgmb import FCGMBOracle

# 1. List available benchmarks (retrieved from internal package configs)
benchmarks = FCGMBOracle.list_benchmarks()
print(f"Available systems: {benchmarks}")

# 2. Instantiate for a specific system
# No need to provide a config path; it uses internal benchmarks by name
oracle = FCGMBOracle("CHEMBL205_1YDA_CHEMBL2331308", budget=5000)

# 3. Get the fragment constraint (SMILES) the model must adhere to
fragment = oracle.fragment
print(f"Target Fragment: {fragment}")

# 4. Get initial set of compounds (lower quartile by pchembl_value from ChEMBL)
initial_df = oracle.get_initial_compounds()

# 5. Score a list of SMILES
# Compounds not matching the fragment OR exceeding budget return 0.0 or NaN
scores = oracle.score(["CCO", "CCC", "CN(C)C"])
# Returns a dict: {SMILES: docking_score}
```

### 2. Zero-Config & Automatic Storage
FCGMB is designed to be run anywhere. It uses a local `.fcgmb` directory for storage:
- **`.fcgmb/grids/`**: Protein grids are prepared once and shared across benchmarks using the same PDB.
- **`.fcgmb/ligand_data/`**: ChEMBL bioactivity data is cached locally to speed up initialization.
- **`.fcgmb/benchmarks/`**: Results and log files for each specific run.

### 3. Docking Workflow
Run the standard docking workflow via the CLI for manual analysis:

```bash
python run_workflow.py --config configs/CHEMBL205_1YDA_CHEMBL2331308.yaml
```

## Directory Structure

- `fcgmb/`: Core package containing logic for docking, receptor preparation, and the Oracle.
  - `configs/`: Bundled benchmark configuration files.
  - `pipeline/`: Scripts for generating benchmark configurations from ChEMBL.
- `.fcgmb/`: (Generated) Local scratch space for grids, cached data, and results.
- `configs/`: (Optional) User-provided benchmark configuration files.
- `notebooks/`: Example notebooks for using the oracle and analyzing results.

## Publication-Ready Variance Figures (SVG)

Use the variance analysis script to generate publication-ready figures with consistent styling:

```bash
python analyze_variance_runs.py \
  --runs-dir variance_runs \
  --config-dir configs \
  --output-dir variance_runs \
  --mapping variance_runs/crystal_ligand_mapping.csv
```

Generated outputs include:
- `variance_runs/system_mean_std_barplot.svg` (plus `.png`)
- `variance_runs/<system_key>_variance_plot.svg` (plus `.png` for each system)

Per-system variance plots include:
- True lower-quartile split using the 25th percentile of `pchembl_value`
- Color coding for model-visible compounds (lower 25%) versus remaining compounds
- Crystal-ligand point annotation when mapping is available

### Crystal-Ligand Mapping CSV

Create `variance_runs/crystal_ligand_mapping.csv` with:

```csv
system_key,molecule_chembl_id,label
CHEMBL204_1MU6_CHEMBL1145961_CHEMBL816574,CHEMBL1145961,Crystal ligand
```

- Required columns: `system_key`, `molecule_chembl_id`
- Optional column: `label` (defaults to `Crystal ligand`)
- If a system mapping is missing or unmatched, the figure is still generated and annotated accordingly.