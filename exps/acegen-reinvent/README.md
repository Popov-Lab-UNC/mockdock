# AceGen-REINVENT × FCGMB

Benchmarks AceGen's **REINVENT** algorithm against all six FCGMB targets using
docking-based fragment-constrained scoring.

## Algorithm

[REINVENT](https://doi.org/10.26434/chemrxiv.14107907.v1) is a likelihood-ratio
policy-gradient algorithm for de novo molecular generation.  It maintains a
frozen prior and an adaptive agent; the agent is trained to maximize a
penalised likelihood objective that balances reward with proximity to the prior.
Experience replay (prioritised) is enabled by default.

## FCGMB adaptations

| Feature | Implementation |
|---------|----------------|
| Fragment conditioning | The benchmark fragment is converted to a PromptSMILES scaffold (single `(*)` attachment point added via RDKit). The model is forced to elaborate the fragment throughout training. |
| Initial compounds | Up to 25 lowest-quartile bioactivity compounds are pre-scored before the RL loop to warm up the docking infrastructure and establish baseline data. |
| Scoring | `FCGMBOracle.score(smiles)` is called each RL iteration; scores are normalised to [0, 1]. |

## Setup

```bash
# Install acegen (editable) and fcgmb into your environment:
pip install -e /work/users/s/h/shuhang/acegen-open
pip install promptsmiles

# fcgmb should already be installed in py312 conda env + acegen venv
```

## Running

```bash
# Submit all 6 benchmarks (default)
sbatch run.sbatch

# Run a single benchmark interactively
python run.py --benchmark AKT1 --budget 5000 --seed 0

# Run a subset of benchmarks
python run.py --benchmark AKT1 --benchmark CHK1 --budget 5000 --seed 0
```

## Outputs

```
outputs/
  <BENCHMARK>/
    initial_compounds_warmup.csv    # pre-scored initial compounds
    reinvent_<BENCHMARK>_<ts>/
      config.yaml                   # exact config used for this run
      compounds.csv                 # all scored SMILES + reward (from acegen Task)
```

## Key hyperparameters

See `config.yaml`. Notable settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sigma` | 120 | REINVENT augmented likelihood weight |
| `num_envs` | 128 | Parallel SMILES generation batch size |
| `experience_replay` | true | Prioritised replay enabled |
| `replay_buffer_size` | 100 | Max molecules in replay buffer |
