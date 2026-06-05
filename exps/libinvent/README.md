# REINVENT4 Lib-INVENT x mockdock

Runs REINVENT4 Lib-INVENT against all six mockdock targets using a custom
REINVENT scoring component that calls `mockdock.MDOracle`.

## Files

- `run.py` - orchestrates benchmark runs, writes per-target TOML, invokes `reinvent`.
- `run.sbatch` - SLURM launcher matching existing `exps` conventions.
- `templates/libinvent_base.toml` - base staged-learning Lib-INVENT template.
- `reinvent_plugins/components/comp_mockdock_oracle.py` - custom scoring bridge.

## Environment

Expected runtime environment:

- REINVENT4 installed and `reinvent` available on PATH.
- mockdock importable (package install or `benchmark/src` on `PYTHONPATH`).
- Docking modules available on cluster (`autodock-gpu`, `autogrid`).
- Lib-INVENT prior file available (default path in `run.sbatch`).

## Quick test

Generate benchmark-specific TOML/config files without launching REINVENT:

```bash
cd /work/users/s/h/shuhang/benchmark/exps/libinvent
python run.py --benchmark DPP4 --budget 50 --dry-run
```

Run one target for real:

```bash
cd /work/users/s/h/shuhang/benchmark/exps/libinvent
python run.py \
  --benchmark DPP4 \
  --budget 50 \
  --prior-file /work/users/s/h/shuhang/REINVENT4/priors/libinvent.prior
```

## Full benchmark suite

```bash
cd /work/users/s/h/shuhang/benchmark/exps/libinvent
sbatch run.sbatch
```

`run.sbatch` runs all six targets (`DPP4 CHK1 ITK PEPCK TTK VEGFR2`) and repeats
the full suite `NUM_RUNS` times into:

- `run_<timestamp>_rXX/<BENCHMARK>/...` for mockdock oracle artifacts.
- `outputs/<BENCHMARK>/...` for TOML/log/checkpoint/summary artifacts.
