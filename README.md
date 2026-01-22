# FCGMB: Fragment-Constrained Generative Model Benchmark

A benchmark package for generative molecular models, specifically using fragment-constrained docking evaluations as a benchmark.

## Installation

```bash
git clone https://github.com/Popov-Lab-UNC/fcgmb.git
cd fcgmb
pip install -e .
```

or using UV:
```bash
uv venv
uv sync
```

## Core Features

### 1. FCGMB Oracle
The `FCGMBOracle` class provides a simple interface for generative models to "score" compounds.

```python
from fcgmb import FCGMBOracle

# Instantiate for a specific system
oracle = FCGMBOracle("CHEMBL205_1YDA_CHEMBL2331308", budget=5000)

# List available systems
benchmarks = FCGMBOracle.list_benchmarks()

# Get initial set of compounds (lower quartile by pchembl_value)
initial_df = oracle.get_initial_compounds()

# Score a list of SMILES
scores = oracle.score(["CCO", "CCC", "CN(C)C"])
# Returns a dict: {SMILES: docking_score}
```

### 2. Docking Workflow
Run the standard docking workflow via the CLI:

```bash
python run_workflow.py --config configs/CHEMBL205_1YDA_CHEMBL2331308.yaml
```

### 3. Variance Testing
Run variance tests to ensure result consistency:

```bash
python run_variance.py --iterations 5
```

## Directory Structure

- `fcgmb/`: Core package containing logic for data, docking, receptor preparation, and the Oracle.
  - `pipeline/`: Scripts for generating benchmark configurations from ChEMBL.
- `configs/`: Curated benchmark configuration files.
- `data/`: Raw and processed data files.
- `benchmarks/`: Output directory for benchmark runs.
- `variance_runs/`: Output directory for variance tests.