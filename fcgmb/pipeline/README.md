# ChEMBL → PDB → Docking Benchmark Pipeline

This pipeline creates a comprehensive benchmark dataset for molecular docking by:
1. Fetching all human protein targets from ChEMBL
2. Mapping them to PDB structures via UniProt
3. Filtering to drug-like co-crystallized ligands
4. Finding ChEMBL documents (patents AND/OR publications) where compounds match the crystal ligand
5. Generating ready-to-use YAML configs for the docking workflow

## Output Files

| File | Description |
|------|-------------|
| `data/chembl_targets.csv` | All human single-protein targets from ChEMBL |
| `data/chembl_pdb_map.csv` | Full ChEMBL → UniProt → PDB → Ligand mapping |
| `data/chembl_pdb_druglike.csv` | Filtered to holo structures with drug-like ligands |
| `data/chembl_docking_benchmark.csv` | Target + Document + PDB combinations with matching ligands |
| `generated_configs/*.yaml` | Ready-to-use docking workflow configs |

## Quick Start

### Option 1: Submit to SLURM (Recommended)
```bash
./run_pipeline.sh slurm
```
This submits all jobs with proper dependencies. Total time: ~4-6 hours.

### Option 2: Run Sequentially
```bash
./run_pipeline.sh sequential
```
Warning: This takes 24+ hours.

### Option 3: Run Individual Steps
```bash
./run_pipeline.sh step1  # Fetch ChEMBL targets (~5 min)
./run_pipeline.sh step2  # Map to PDB (array job, ~2 hours total)
./run_pipeline.sh step3  # Filter drug-like (~30 min)
./run_pipeline.sh step4  # Find matching documents (array job, ~4 hours total)
./run_pipeline.sh step5  # Generate configs (~5 min)
```

## Script Details

### 01_fetch_chembl_targets.py
Fetches all human single-protein targets from ChEMBL with their UniProt accessions.

### 02_map_pdb_ligands.py
For each UniProt ID, queries RCSB PDB for:
- All PDB structures containing that protein
- Co-crystallized ligand information (residue name, SMILES)
- Resolution information

**Parallelization**: Use `--start` and `--end` flags to process subsets.

### 03_filter_druglike.py
Filters ligands using RDKit descriptors:
- Molecular weight: 200-800 Da
- At least one ring
- Contains nitrogen or oxygen
- LogP: -2 to 7
- Rotatable bonds: ≤ 15

### 04_find_matching_documents.py
For each target with drug-like crystal ligands:
1. Fetches all ChEMBL documents (patents and/or publications)
2. Computes fingerprint similarity between crystal ligand and document compounds
3. Reports matches above the similarity threshold

**Key Options**:
- `--include-patents` / `--no-patents`: Toggle patent inclusion
- `--include-publications` / `--no-publications`: Toggle publication inclusion
- `--similarity-threshold`: Tanimoto threshold (default: 0.7)
- `--min-compounds`: Minimum compounds in document to consider (default: 20)

**Parallelization**: Use `--start` and `--end` flags to process subsets.

### 05_generate_benchmark_configs.py
Generates YAML workflow configs for the docking pipeline, including:
- Murcko scaffold extraction for fragment constraint
- Resolution-based filtering
- Document type annotation

### merge_results.py
Utility to merge partial results from parallel SLURM jobs.

## Customization

### To only use patents:
```bash
python scripts/04_find_matching_documents.py \
    --include-patents \
    --no-publications \
    ...
```

### To only use publications:
```bash
python scripts/04_find_matching_documents.py \
    --no-patents \
    --include-publications \
    ...
```

### To adjust similarity threshold:
```bash
python scripts/04_find_matching_documents.py \
    --similarity-threshold 0.8 \
    ...
```

## Monitoring

Check SLURM job status:
```bash
squeue -u $USER
```

Check logs:
```bash
tail -f logs/02_pdb_map_*.log
```
