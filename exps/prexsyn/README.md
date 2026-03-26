# PrexSyn × FCGMB Evaluation

Evaluates [PrexSyn](https://arxiv.org/abs/2512.00384) on the six FCGMB docking
benchmarks. The script mirrors the structure of
`prexsyn/scripts/benchmarks/optim.py` as closely as possible, using the same
`Task` / `Optimizer` / `FingerprintGenetic` / `CachedOracle` / `OptimTracker`
building blocks from PrexSyn. The only FCGMB-specific piece is
`FCGMBOracleAdapter`, which adapts `FCGMBOracle.score()` to PrexSyn's
`OracleProtocol`.

## Script

### `run.py`

PrexSyn's `Optimizer` (with `FingerprintGenetic` step strategy) drives the
loop. The FCGMB oracle call is isolated in `FCGMBOracleAdapter.__call__()`,
which makes the SMILES list → score dict exchange visible:

```python
score_map: dict[str, float] = self._oracle.score(smiles_list)
```

### Outputs (per benchmark)

```
outputs/<BENCHMARK>/
├── run_01.df.pkl      # OptimTracker DataFrame (pickle) — same as prexsyn optim.py
├── log.txt            # Per-benchmark log
└── oracle_results.csv # Full docking records from FCGMBOracle
```

## Usage

```bash
# Run all six benchmarks (default)
python run.py --model /path/to/v1_converted.yaml

# Single benchmark
python run.py --benchmark CHK1 --budget 5000

# With soft BRICS fragment conditioning
python run.py --benchmark CHK1 --fragment-condition

# Multiple benchmarks, 3 independent runs each
python run.py --benchmark DPP4 --benchmark CHK1 --num-runs 3

# Full options
python run.py \
    --benchmark      CHK1 \
    --model          /path/to/prexsyn/data/trained_models/v1_converted.yaml \
    --budget         5000 \
    --num-runs       1 \
    --num-init-samples 500 \
    --bottleneck-size 50 \
    --time-limit     86400 \
    --fragment-condition \
    --out            ./outputs
```

## Fragment Conditioning

PrexSyn conditions generation on **BRICS fragment fingerprints** via its
`BRICSCondition` / `QueryConditionedSampler` machinery.  This is a **soft bias**:
the model is steered toward molecules that decompose into fragments resembling the
target fragment, but there is no hard guarantee the fragment will appear.

Hard fragment enforcement is handled by `FCGMBOracle` — molecules that fail the
2D substructure check get score `0.0` and **do not count against the budget**.

In addition, this benchmark wrapper can use FCGMB initial compounds as context
(`--initial-context`, enabled by default). It builds an ECFP reference query
from selected initial compounds and combines it with fragment conditioning when
`--fragment-condition` is used.

You can therefore run with or without `--fragment-condition` and compare:
- **Without**: PrexSyn explores drug-like space freely; oracle filters enforce fragment.
- **With**: PrexSyn is biased toward fragment-containing analogues during generation.

## Dependencies

Both packages must be installed in the same environment:

```bash
pip install -e /path/to/prexsyn
pip install -e /path/to/benchmark   # or pip install fcgmb
```
