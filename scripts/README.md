# Benchmark Scripts

Scripts are organized by workflow stage. Run commands from the repository root unless a script says otherwise.

| Stage | Directory | Purpose |
| --- | --- | --- |
| **Dataset Generation** | `scripts/dataset/` | Data mining pipeline from ChEMBL SQLite and PDB to build benchmark candidates and configs. |
| **Docking Workflows** | `scripts/docking/` | Core AutoDock-GPU docking workflow executed locally or dispatched via SLURM. |
| **Variance Calibration** | `scripts/variance/` | Multi-replicate docking variance calibration, baseline scoring, and stability analysis. |
| **Orthogonal Scoring** | `scripts/scoring/` | MolSkill (medicinal chemist preference), STOPLIGHT (ADMET liabilities), and AIZynthFinder (retrosynthetic feasibility) scoring pipelines. |
| **Analysis & Plotting** | `scripts/analysis/` | Compute benchmark metrics, property correlations, chemical quality statistical tests, and generate publication figures. |
| **Validation** | `scripts/validation/` | End-to-end benchmark validation and docking integrity checks across configured targets. |
| **Orchestration** | `scripts/orchestration/` | Local execution or SLURM job-dependency pipelines running analysis and scoring end to end. |
| **Common Helpers** | `scripts/common/` | Shared shell environment configuration (`env.sh`, `slurm.sh`) and experiment utilities. |

---

## Environments & Setup

Most scripts expect the virtual environment at `.venv` and source [`scripts/common/env.sh`](file:///hickory/users/s/h/shuhang/devel/mockdock/scripts/common/env.sh), which exports `BENCHMARK_DIR` and prepends `src/` to `PYTHONPATH`.

Scoring modules utilize specialized environments where required:
- **MolSkill**: Conda environment (`molskill`) via `mockdock_activate_conda molskill`.
- **AIZynthFinder**: Python virtual environment containing AIZynthFinder models and stock database policies.
- **STOPLIGHT**: Communicates via persistent worker daemons (`stoplight_worker_daemon.py`).

Dataset extraction scripts accept `CHEMBL_SQLITE_PATH=/path/to/chembl_36.db`.

---

## Common Workflows & Commands

### 1. Dataset Generation Pipeline
```bash
# Sequential local run
export CHEMBL_SQLITE_PATH=/path/to/chembl_36.db
bash scripts/dataset/run_dataset_pipeline.sh sequential

# Or SLURM array dispatch on cluster
bash scripts/dataset/run_dataset_pipeline.sh slurm
```

### 2. Docking Execution & Validation
```bash
# Run validation across benchmark targets
python scripts/validation/validation_test.py

# Submit batch docking workflow via SLURM
sbatch scripts/docking/run_workflow_longleaf.sbatch
```

### 3. Variance Calibration
```bash
# Run 5-replicate variance calibration for a target
python scripts/variance/run_variance.py \
    --config src/mockdock/configs/PptT.toml \
    --run-dir variance_runs/PptT \
    --output-dir variance_analysis/PptT \
    --n-iters 5

# Or submit via SLURM
sbatch scripts/variance/slurm_pptt_variance.sbatch
```

### 4. Orthogonal Scoring (MolSkill, STOPLIGHT, AIZynthFinder)
```bash
# Score generated molecules with MolSkill
python scripts/scoring/score_molecules.py --scorer molskill --exps-dir exps_upperbound --batch-size 64

# Score retrosynthetic feasibility with AIZynthFinder
python scripts/scoring/score_molecules.py --scorer aizynthfinder --exps-dir exps_upperbound

# Score ADMET liabilities with STOPLIGHT
python scripts/scoring/score_molecules.py --scorer stoplight --exps-dir exps_upperbound
```

### 5. Analysis & Metric Aggregation
```bash
# Aggregate metrics across all models and targets
python scripts/analysis/analyze_experiments.py \
    --exps-dir exps_upperbound \
    --output-dir analysis_exps_upperbound

# Calculate chemical quality statistical distributions vs. reference set
python scripts/analysis/calculate_chemical_quality_distribution_tests.py \
    --exps-dir exps_upperbound \
    --output-dir analysis_exps_upperbound

# Generate property correlation analysis (MW / cLogP vs docking score)
python scripts/analysis/correlation_analysis.py \
    --exps-dir exps_upperbound \
    --output-dir assets/correlation

# Convert an AutoDock .dlg output pose to .sdf
python scripts/analysis/convert_dlg_to_sdf.py -i pose.dlg -o pose.sdf --pose-index 0
```

### 6. Pipeline Orchestration
```bash
# Run full analysis and scoring sequentially
bash scripts/orchestration/run_analysis_scoring_pipeline.sh

# Or submit with chained SLURM job dependencies
bash scripts/orchestration/submit_analysis_scoring_pipeline.sh
```
