# AceGen-A2C × FCGMB

Benchmarks AceGen's **A2C** (Advantage Actor-Critic) algorithm against all six
FCGMB targets.

## Algorithm

A2C is a synchronous actor-critic algorithm using GAE advantage estimation.
AceGen's implementation uses shared actor/critic networks.  A2C does **not**
include an experience replay buffer in AceGen's standard implementation.

## FCGMB adaptations

| Feature | Implementation |
|---------|----------------|
| Fragment conditioning | Benchmark fragment → PromptSMILES scaffold (single `(*)` attachment point). Fragment conditioning is the primary form of domain knowledge for A2C, since there is no replay buffer to pre-seed. |
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
| `num_envs` | 16 |
| `gamma` | 0.999 |
| `entropy_coef` | 0.05 |
| `critic_coef` | 0.5 |
