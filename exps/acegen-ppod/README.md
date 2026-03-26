# AceGen-PPOD × FCGMB

Benchmarks AceGen's **PPOD** algorithm against all six FCGMB targets.

## Algorithm

PPOD is PPO with a prioritised **experience replay buffer** enabled
(`experience_replay: true`).  The replay buffer retains high-reward previously
generated molecules and mixes them with fresh on-policy data at each update,
improving sample efficiency and reducing mode collapse.

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
python run.py --benchmark DPP4 --budget 5000 --seed 0
```

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| `ppo_clip` | 0.5 |
| `ppo_epochs` | 3 |
| `num_envs` | 64 |
| `experience_replay` | **true** |
| `replay_buffer_size` | 100 |
