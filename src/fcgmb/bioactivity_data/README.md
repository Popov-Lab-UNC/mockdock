# Bioactivity Data Directory

This directory contains bundled ground-truth bioactivity data for each benchmark.

## File Naming Convention

Files are named by `benchmark_name` as defined in the corresponding config YAML:

```
bioactivity_data/
├── AKT1.csv
├── CHK1.csv
├── ITK.csv
├── PCK1.csv
├── TTK.csv
└── VEGFR2.csv
```

## CSV Format

| Column              | Description                                     |
|---------------------|-------------------------------------------------|
| molecule_chembl_id  | ChEMBL compound identifier                      |
| canonical_smiles    | Standardized/canonicalized SMILES string        |
| pchembl_value       | −log₁₀(IC₅₀/Ki/Kd) activity value             |

## Data Splitting

The oracle computes the lower 25% of the pChEMBL range as the **initial set** 
(returned by `get_initial_compounds()`), and the remaining upper 75% as the 
**validation set** (returned by `get_validation_compounds()`).

## Adding Your Own Benchmark

1. Name your CSV `<benchmark_name>.csv` matching the `benchmark_name` field in your config.
2. Ensure columns `molecule_chembl_id`, `canonical_smiles`, and `pchembl_value` are present.
3. Place the CSV in this directory.

If no bundled CSV is found, the oracle will fall back to a local scratch cache
at `.fcgmb/data/<benchmark_name>_chembl.csv`, and ultimately to a live ChEMBL API call.

## Regenerating This Data

Run `fetch_bioactivity.py` from the benchmark root:

```bash
cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate
python fetch_bioactivity.py
```
