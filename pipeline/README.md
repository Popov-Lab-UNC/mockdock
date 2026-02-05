# FCGMB Dataset Generation Pipeline

This pipeline generates the **FCGMB (Fragment-Constrained Generative Molecular Benchmark)** dataset. It mines ChEMBL and PDB to find co-crystallized ligands that have associated bioactivity data in ChEMBL documents, automatically labeling them as 'literature' or 'other'.

## 🚀 Key Features

*   **Assay-Level Granularity**: Matches are grouped by specific ChEMBL assays (`assay_chembl_id`), ensuring bioactivity data (IC50/Ki/etc.) comes from consistent experimental conditions.
*   **Unified ChEMBL Caching**: Minimizes API calls by caching raw bioactivity data locally (`data/chembl_cache/`), organized by target and assay.
*   **Robust MCS Calculation**: Computes the Maximum Common Substructure (MCS) specific to the compounds in each assay to generate accurate fragment constraints.
*   **SMILES Standardization**: Automatically standardizes and neutralizes molecules before processing to ensure high-quality matching.
*   **Resilience**: Includes API health checks (`probe_chembl.py`) and granular checkpointing.

## 📂 Directory Structure

*   `pipeline/`: Contains all scripts for **generating** the benchmark dataset.
*   `fcgmb/`: The core python package for **running** the benchmarks (imports generated data).
*   `data/`: Working directory for intermediate and final outputs.

## 🛠️ Usage

### Quick Start (SLURM)
The recommended way to run the full pipeline on a cluster. Includes dependency management (Step 2 starts after Step 1, etc.).

```bash
export CHEMBL_SQLITE_PATH=/path/to/chembl_36.db
bash pipeline/run_pipeline.sh slurm
```

### Sequential Run
For small tests or local machines (warning: full run is very slow).
```bash
export CHEMBL_SQLITE_PATH=/path/to/chembl_36.db
bash pipeline/run_pipeline.sh sequential
```

### Manual Steps
You can run individual steps manually:
```bash
python pipeline/probe_chembl.py  # Check if ChEMBL is up
python pipeline/fetch_chembl_targets.py --chembl-sqlite /path/to/chembl_36.db
python pipeline/map_pdb_ligands.py
# ... etc
```

## 📜 Script Reference & Defaults

### 1. Data Retrieval & Mapping

#### `probe_chembl.py`
*   **Purpose**: A diagnostic tool to check if the ChEMBL API is currently reachable.

#### `fetch_chembl_targets.py` (Step 1)
*   **Purpose**: Fetches all human single-protein targets.
*   **Default Output**: `data/chembl_targets.csv`
*   **SQLite**: Pass `--chembl-sqlite /path/to/chembl_36.db` or set `CHEMBL_SQLITE_PATH`.

#### `map_pdb_ligands.py` (Step 2)
*   **Purpose**: Maps targets to PDB structures and co-crystallized ligands.
*   **Key Arguments & Defaults**:
    *   `--delay 0.1`: Seconds to sleep between PDB API calls to be polite.

#### `filter_druglike.py` (Step 3)
*   **Purpose**: Filters ligands based on molecular properties.
*   **Logic**:
    *   MW: 200 - 800 Da
    *   Rings: >= 1
    *   Heteroatoms: Must contain N or O
    *   LogP: -2 to 7
    *   Rotatable Bonds: <= 15

### 2. Mining Bioactivity

#### `find_matching_documents.py` (Step 4)
*   **Purpose**: Finds ChEMBL assays containing the crystal ligand series.
*   **Key Arguments & Defaults**:
    *   `--similarity-threshold 1.0`: **Default is Exact Match**. Tanimoto similarity threshold (Morgan FP, radius 2) to consider a compound "matching" the crystal ligand. Lower to 0.7-0.9 to find analogs.
    *   `--min-compounds 20`: Minimum number of compounds required in an assay to include it. Data-rich assays are preferred for benchmarking.

    **Example: Relaxing the constraint**
    ```bash
    # Allow close analogs (0.8 similarity) and smaller assays (10 compounds)
    python pipeline/find_matching_documents.py \
        --similarity-threshold 0.8 \
        --min-compounds 10 \
        --chembl-sqlite /path/to/chembl_36.db
    ```

#### `compute_mcs.py` (Step 4b)
*   **Purpose**: Computes Maximum Common Substructure for each assay.
*   **Key Arguments & Defaults**:
    *   `--max-compounds 100`: Max compounds per assay to use for MCS calculation (for speed).
    *   `--timeout 10`: Seconds before timing out MCS calculation for a single assay.

### 3. Configuration Generation

#### `generate_benchmark_configs.py` (Step 5)
*   **Purpose**: Generates final YAML configs.
*   **Key Arguments & Defaults**:
    *   `--min-compounds 20`: **Stricter filter**. Only generates configs for assays with at least 20 compounds (ensures robust stats).
    *   `--top-n 100`: Limits output to the top 100 entries sorted by resolution (best structures first). Set to 0 for all.
    *   `--require-crystal-in-assay` (Default: True): Requires the crystal ligand itself (or exact match) to be present in the assay.

    **Example: Generating all possible benchmarks**
    ```bash
    python pipeline/generate_benchmark_configs.py --top-n 0 --min-compounds 10
    ```

## 📊 Outputs

| File | Description |
|------|-------------|
| `data/chembl_docking_benchmark.csv` | The master index of Target-PDB-Assay combinations. |
| `data/mcs_results.csv` | Computed MCS SMARTs/SMILES for each assay. |
| `data/chembl_cache/` | Directory containing raw bioactivity data CSVs, organized by `{target_id}/{doc_id}_{assay_id}.csv`. |
| `generated_configs/` | Final YAML files to be deployed to `fcgmb/configs/` for use. |
