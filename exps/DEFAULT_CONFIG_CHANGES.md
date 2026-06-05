# Default Config Changes

Tracks intentional changes to benchmark experiment defaults that affect stopping behavior and scoring semantics.

## 2026-04-11

### Lib-INVENT (`benchmark/exps/libinvent/templates/libinvent_base.toml`)

- Changed stage termination:
  - `termination = "simple"` -> `termination = "null"`
- Kept:
  - `max_score = 1.0` (now informational while `termination = "null"`)

Impact:
- REINVENT4 no longer early-stops when stage score reaches `max_score`.
- Stage runs until `max_steps` is reached.

### PrexSyn (`benchmark/exps/prexsyn/run.py`, `prexsyn/prexsyn/applications/optim/optim.py`)

- Disabled oracle result caching for mockdock scoring path:
  - `CachedOracle(MDOracleAdapter(...))` -> `MDOracleAdapter(...)`
- Added optimizer stop hook:
  - `stop_condition=lambda: self.md_oracle.status != "active"`
- Added `stop_condition` support to `Optimizer` and loop check in `Optimizer.run()`.

Impact:
- Every generated molecule submitted in optimization is explicitly scored by MDOracle.
- Duplicate molecules are no longer skipped by cache in this benchmark path.
- Run stops when oracle budget is exhausted (or `time_limit` if set), instead of relying only on unique-tracker length.

## Notes

- No package reinstall is required for these changes in the current setup, because benchmark scripts import from local source trees (including `prexsyn` via `PYTHONPATH` in `benchmark/exps/prexsyn/run.sbatch`).

## 2026-04-11 (results accounting)

### mockdock Oracle (`benchmark/src/mockdock/oracle.py`)

- Updated `MDOracle.score()` exhausted-budget branch to append `budget_exhausted`
  rows to `results.csv`/`results.yaml` for all post-budget model emissions.
- Added `generation_index` (monotonic per-emission row index) to output rows
  and included it in `results.yaml`.

Impact:
- `results.csv` now remains a complete per-emission record even when model code
  keeps calling the oracle after the budget is already exhausted.
- `results.yaml` now includes both `generation_round` and `generation_index`,
  so entries can be traced back to exact emission position.

## 2026-04-16

### PrexSyn — full rewrite of `benchmark/exps/prexsyn/run.py`

The prexsyn package was refactored upstream and the entire `prexsyn.applications.optim`
API (`Optimizer`, `FingerprintGenetic`, `OptimTracker`) no longer exists. `run.py` was
completely rewritten to use the current API.

**API migration:**

| Old (broken) | New |
|---|---|
| `from prexsyn.applications.optim import Optimizer` | `from prexsyn.shortcuts.genetic import initialize, evolve` |
| `from prexsyn.factories import load_model` | `from prexsyn.shortcuts import AllInOneLoader` |
| `FingerprintGenetic(bottleneck_size=...)` | `evolve(ppl, history, projector, fn, k=..., t=...)` |
| `OptimTracker` | Custom `list[dict]` → `pd.DataFrame` tracker |
| `Facade` / `PropertySet` / `Query` | Removed (not used in oracle-eval path) |

**Anti-cheat — deduplication budget charging (`MDOracleAdapter`):**

- PrexSyn's internal `Population.dedup()` removes repeated SMILES from the
  elite population before calling the oracle, making repeated molecules free.
- `MDOracleAdapter` now tracks all SMILES it has ever scored in `_seen`.
- When a repeated SMILES appears in a batch, the cached score is returned but
  `oracle.budget_used` is incremented directly — same cost as a novel molecule.

**Anti-cheat — removed early stopping:**

- Old `stop_condition=lambda: self.md_oracle.status != "active"` could trigger
  an internal early-stop (including at score 1.0).
- The optimization loop is now implemented directly (`while not budget_exhausted`)
  with no early-stop condition. Run terminates only when `max_budget` is reached
  (or `time_limit` if set).

**Context Initialization (`initialize_from_context`):**

- PrexSyn's default `initialize()` naturally starts from entirely random `ecfp4` fingerprints.
- Created `initialize_from_context` which intercepts `md_oracle.get_initial_compounds()`, computes their ECFP4 fingerprints via the `MoleculeProjector`, and seeds the initial genetic `EmbryoSet` with a soft 1% baseline mutation to anchor early exploration around the target's baseline actives.

### PrexSyn — environment migration (`benchmark/exps/prexsyn/run.sbatch`)

### All Scripts — GPU Partition Update

- Globally updated `run.sbatch` SLURM scripts across all models to use the `--partition=l40-gpu` instead of `a100-gpu`, presumably to optimize resource usage on the cluster.

## 2026-04-21

### Lib-INVENT (`benchmark/exps/libinvent/templates/libinvent_base.toml`)

- Disabled sequence-level deduplication at generation time:
  - `unique_sequences = true` -> `unique_sequences = false`

Impact:
- REINVENT4 no longer drops duplicate generated sequences before oracle scoring.
- Effective oracle consumption per step increases, so runs are far more likely to
  reach the configured budget instead of stopping below budget due to dedup loss.

### Lib-INVENT mockdock scoring plugin (`benchmark/exps/libinvent/reinvent_plugins/components/comp_mockdock_oracle.py`)

- Added generation-time accounting in `MockdockOracle`:
  - track inter-call elapsed time with `time.perf_counter()`
  - accumulate generated ligand count from each `__call__(smiles)` batch
- Passed both values into `MDOracle.save_metrics(...)` as:
  - `total_generation_time_sec`
  - `n_generated_ligands`

Impact:
- `metrics.json` now reports non-zero generation metrics (`total_gen_time`,
  `avg_gen_time_per_mol`) for Lib-INVENT runs, rather than defaulting to `0.0`.

## 2026-04-28

### Lib-INVENT mockdock scoring plugin (`benchmark/exps/libinvent/reinvent_plugins/components/comp_mockdock_oracle.py`)

- Changed budget accounting semantics in `MockdockOracle.__call__`:
  - every generated SMILES now consumes budget, including duplicates
  - oracle scoring is truncated to remaining budget (`smiles[:remaining_budget]`)
  - overflow emissions receive score `0.0` and are still counted toward consumed budget
- Added post-run budget reconciliation in `MockdockOracle._rewrite_budget_accounting()`:
  - rewrites `status.json` and `metrics.json` so `budget_used` reflects generated-SMILES consumption
  - updates `n_molecules_total` to match this consumed budget
- Added explicit metric field:
  - `consumed_budget_generated_smiles`

Impact:
- Budget consumption is now anti-cheat with respect to duplicate emissions: repeated SMILES are no longer "free" for Lib-INVENT runs.
- Reported `budget_used` now matches the per-emission accounting rule used in this benchmark path.

### Lib-INVENT runner + stop behavior (`benchmark/exps/libinvent/run.py`, `benchmark/exps/libinvent/reinvent_plugins/components/comp_mockdock_oracle.py`)

- Switched stage-step default from budget-derived cap to an intentionally huge cap:
  - `max_steps = ceil(budget / 64)` -> `max_steps = max(1_000_000, ceil(budget / 64))`
- Added explicit budget-stop path in `MockdockOracle.__call__`:
  - when consumed budget reaches configured budget, finalize outputs and raise `BudgetExhaustedStop`
  - write `budget_exhausted.json` marker in benchmark run directory
- Updated runner subprocess handling in `_run_reinvent(...)`:
  - non-zero REINVENT exit is treated as successful termination if `budget_exhausted.json` exists
  - otherwise non-zero exit remains a hard failure

Impact:
- Lib-INVENT stopping is now budget-driven by default rather than step-cap-driven.
- Premature termination due to `max_steps` should no longer occur in normal runs.
- Existing failure semantics are preserved for real errors unrelated to budget exhaustion.

## 2026-04-29

### MockDock scoring semantics (`benchmark/src/mockdock/oracle.py`, `benchmark/src/mockdock/evaluator.py`)

- Split score outputs into three explicit columns:
  - `docking_score`: raw docking energy, lower is better.
  - `norm_score`: uncapped normalized score.
  - `reward_score`: `norm_score` clipped to `[0, 1]`.
- Kept `normalized_score` as a backward-compatible alias for `reward_score`.
- Skipped/failed compounds now receive `0.0` reward instead of `-1.5`.
- `MDOracle.score()` returns `reward_score`, so RL algorithms learn from a
  bounded reward.

Impact:
- Avoids negative or unbounded rewards destabilizing RL algorithms.
- Preserves uncapped docking improvement in `norm_score` for post-hoc analysis.

### MockDock calibration (`benchmark/src/mockdock/configs/*.yaml`, `benchmark/src/mockdock/bioactivity_data/*.csv`)

- Recomputed `low_score` and `high_score` from
  `/work/users/s/h/shuhang/mockdock_data/variance_runs`:
  - `high_score`: best/minimum mean docking score across original benchmark compounds.
  - `low_score`: worst/maximum mean docking score across original benchmark compounds.
- Refreshed benchmark bioactivity CSVs with `mean_docking_score`, `norm_score`,
  `reward_score`, and compatibility `score`.

Impact:
- A reward of `1.0` now means matching or exceeding the best original benchmark
  docking score for that target.
- Scores above the original benchmark range remain visible through `norm_score`
  but no longer increase the RL reward beyond `1.0`.

## 2026-04-30

### AceGen batch-size standardization (`benchmark/exps_oldscoring_medchem/acegen-*/config.yaml`)

- Standardized effective generation batch size across AceGen experiment templates:
  - `acegen-a2c`: `num_envs: 16` -> `num_envs: 64`
  - `acegen-ahc`: `num_envs: 128` -> `num_envs: 64`
  - `acegen-reinforce`: `num_envs: 128` -> `num_envs: 64`
  - `acegen-reinvent`: `num_envs: 128` -> `num_envs: 64`
- `acegen-ppo` and `acegen-ppod` were already configured with `num_envs: 64`.

Impact:
- AceGen models now use the same effective per-round generation batch size in
  future benchmark runs.
- This makes `avg_gen_time_per_mol` comparisons less confounded by batch-size
  differences, and aligns AceGen PPO/PPOD-style generation with Lib-INVENT's
  `batch_size = 64` setting.

## 2026-05-03

### Lib-INVENT reward stabilization (`benchmark/exps/libinvent/reinvent_plugins/components/comp_mockdock_oracle.py`)

- Added RL-facing reward mapping in `MockdockOracle`:
  - new `_to_rl_reward(raw_score)` helper
  - when `clip_reward_upper_bound = true`: keep hard cap behavior (`min(max(raw, 0), 1)`)
  - when `clip_reward_upper_bound = false`: apply smooth bounded transform
    `raw / (1 + raw)` after floor at `0.0`
- Applied mapping in `MockdockOracle.__call__` before returning `ComponentResults`
  to REINVENT.

Impact:
- Prevents runaway reward magnitude when uncapped oracle scores exceed `1.0`,
  which was causing unstable DAP policy updates and rapid mode collapse.
- Avoids collapse-driven duplicate-heavy sampling that made Lib-INVENT appear to
  generate only ~1 unique molecule per round (massive slowdown in practice).
- Preserves score ordering for learning and keeps unclipped oracle metrics
  available in run artifacts for analysis.

## 2026-05-26

### Standard REINVENT (`benchmark/exps/reinvent/`)

- Created a new experiment suite for standard REINVENT4 staged-learning *de novo* molecular design.
- Implemented `warmup_oracle` to pre-score up to `n_warmup` (default `25`) initial bioactivity compounds.
- Added dynamic `[inception]` memory configuration generated from warmed-up initial compound SMILES to guide policy gradient learning.
- Stripped Stereochemistry for Inception: Automatically strips stereochemical and chiral descriptors (`[C@H]`, `[C@@H]`, `/`, `\`) from warmup active compounds written to `initial_compounds.smi`. This prevents REINVENT4 validation failures caused by chiral tokens not present in the model's standard `reinvent.prior` vocabulary.
- Standardized effective generation batch size to `batch_size = 64`.
- Enabled support for dual scoring caps (clipped to `[0, 1]` or unbounded using smooth `score / (1 + score)` scaling).

### Lib-INVENT (`benchmark/exps/libinvent/`)

- Updated `libinvent/run.py` to support `warmup_oracle` and the `--n-warmup` (default `25`) option.
- Enabled reuse of pre-computed initial compound docking scores, matching the efficiency and baseline recording of `acegen-*` and standard `reinvent` runs.
- Updated `libinvent/run.sbatch` to explicitly specify and pass `--n-warmup 25`.

