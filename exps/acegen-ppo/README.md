# AceGen-PPO × FCGMB

Benchmarks AceGen's **PPO** algorithm against all six FCGMB targets using
docking-based fragment-constrained scoring.

## Algorithm

[PPO (Proximal Policy Optimisation)](https://arxiv.org/abs/1707.06347) is an
actor-critic on-policy RL algorithm.  AceGen's implementation uses a shared
GRU backbone with a separate critic head, optimised with clipped surrogate loss
and GAE advantage estimation.  Experience replay is **disabled** in this folder
(see `acegen-ppod` for PPO+D with replay).

## FCGMB adaptations

| Feature | Implementation |
|---------|----------------|
| Fragment conditioning | The benchmark fragment is converted to a PromptSMILES scaffold (single `(*)` attachment point added via RDKit). The model is forced to elaborate the fragment throughout training. |
| Initial compounds | Up to 25 lowest-quartile bioactivity compounds are pre-scored before the RL loop to warm up the docking infrastructure and establish baseline data. |
| Scoring | `FCGMBOracle.score(smiles)` is called each RL iteration; scores are normalised to [0, 1]. |

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

## Outputs

```
outputs/
  <BENCHMARK>/
    initial_compounds_warmup.csv
    ppo_<BENCHMARK>_<ts>/
      config.yaml
```

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| `ppo_clip` | 0.5 |
| `ppo_epochs` | 3 |
| `num_envs` | 64 |
| `experience_replay` | false |
