# AceGen-AHC × mockdock

Benchmarks AceGen's **AHC** (Augmented Hill-Climbing) algorithm against all
six mockdock targets.

## Algorithm

AHC is a REINVENT-variant that selects only the top-k fraction of generated
molecules (by score) to compute the training loss, effectively performing hill
climbing in chemical space.  Prioritised experience replay is enabled.

## mockdock adaptations

| Feature | Implementation |
|---------|----------------|
| Fragment conditioning | Benchmark fragment → PromptSMILES scaffold (single `(*)` attachment point). |
| Initial compounds | Up to 25 lowest-quartile compounds pre-scored for oracle warmup. |
| Scoring | `mockdockOracle.score(smiles)`, normalised [0, 1]. |

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
| `topk` | 0.5 (top 50 % of batch) |
| `sigma` | 60 |
| `num_envs` | 128 |
| `experience_replay` | true |
