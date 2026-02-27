# AceGen-REINFORCE × FCGMB

Benchmarks AceGen's **REINFORCE** algorithm against all six FCGMB targets.

## Algorithm

REINFORCE is a simple policy-gradient algorithm.  AceGen's implementation
includes an optional prioritised experience replay buffer.

## FCGMB adaptations

| Feature | Implementation |
|---------|----------------|
| Fragment conditioning | Benchmark fragment → PromptSMILES scaffold (single `(*)` attachment point). |
| Initial compounds | Up to 25 lowest-quartile compounds pre-scored for oracle warmup. |
| Scoring | `FCGMBOracle.score(smiles)`, normalised [0, 1]. |

## Setup

```bash
pip install -e /work/users/s/h/shuhang/acegen-open
pip install promptsmiles
```

## Running

```bash
sbatch run.sbatch
python run.py --benchmark AKT1 --budget 5000 --seed 0
```

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| `num_envs` | 128 |
| `lr` | 0.0001 |
| `experience_replay` | true |
| `replay_buffer_size` | 100 |
