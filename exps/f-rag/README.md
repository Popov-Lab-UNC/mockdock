# f-RAG x FCGMB Evaluation

This experiment adapts the original [f-RAG](https://arxiv.org/abs/2411.12078)
generation loop to FCGMB docking benchmarks.

## What stays the same as f-RAG

- SAFE generation + GA reproduction loop
- Arm / linker fragment populations
- Retrieval-augmented generation through `SAFEFusionDesign`

## What changes for FCGMB

- Uses `FCGMBOracle` as the scoring oracle.
- Uses `oracle.get_initial_compounds()` as the starting context.
  - Initial compounds seed both:
    - molecule population (`mol_population`)
    - arm/linker fragment populations used for retrieval.

This explicitly uses benchmark-provided initial compounds as requested by FCGMB.

## Files

- `run.py`: main runner
- `hparams.yaml`: defaults matching original f-RAG docking setup

## Usage

```bash
cd benchmark/exps/f-rag

# Run all six FCGMB benchmarks
python run.py --budget 5000 --seed 0

# Single benchmark
python run.py --benchmark CHK1 --budget 5000

# Optional: provide local checkout path if modules are not importable
python run.py --f-rag-root ../../../f-RAG --benchmark CHK1 --budget 5000
```

Outputs are written under `outputs/<BENCHMARK>/`:

- `seed_<N>.csv`: generated molecules with FCGMB score
- `oracle_results_seed_<N>.csv`: full docking records from `FCGMBOracle`
- `log.txt`: run log
