# mockdock Dataset Generation Pipeline

This pipeline generates the mockdock fragment-constrained generative molecular benchmark dataset. It mines ChEMBL and PDB for co-crystallized ligands with associated bioactivity data and labels sources as literature or other.

## Directory Structure

- `scripts/dataset/`: Dataset-generation scripts and SLURM launchers.
- `scripts/common/`: Shared environment helpers used by launchers.
- `src/mockdock/`: Runtime package that consumes generated benchmark configs and data.
- `data/`: Intermediate and final dataset outputs.

## Usage

Run from the repository root.

```bash
export CHEMBL_SQLITE_PATH=/path/to/chembl_36.db
bash scripts/dataset/run_dataset_pipeline.sh slurm
```

For a sequential local run:

```bash
export CHEMBL_SQLITE_PATH=/path/to/chembl_36.db
bash scripts/dataset/run_dataset_pipeline.sh sequential
```

Manual steps use the same stage paths:

```bash
python scripts/dataset/probe_chembl.py
python scripts/dataset/fetch_chembl_targets.py --chembl-sqlite /path/to/chembl_36.db
python scripts/dataset/map_pdb_ligands.py
python scripts/dataset/filter_druglike.py
python scripts/dataset/find_matching_documents.py --chembl-sqlite /path/to/chembl_36.db
python scripts/dataset/compute_mcs.py
python scripts/dataset/generate_benchmark_configs.py
```

## Pipeline Steps

1. `fetch_chembl_targets.py`: Fetch human single-protein ChEMBL targets.
2. `map_pdb_ligands.py`: Map ChEMBL targets to PDB structures and ligands.
3. `filter_druglike.py`: Keep drug-like ligands.
4. `find_matching_documents.py`: Find ChEMBL assays/documents matching crystal ligand series.
5. `compute_mcs.py`: Compute maximum common substructures for matching assays.
6. `generate_benchmark_configs.py`: Generate benchmark YAML configs.

## Outputs

- `data/chembl_docking_benchmark.csv`: Master target/PDB/assay index.
- `data/mcs_results.csv`: MCS SMARTS/SMILES results.
- `data/chembl_cache/`: Cached ChEMBL bioactivity data.
- `generated_configs/`: Generated configs for `src/mockdock/configs/`.
