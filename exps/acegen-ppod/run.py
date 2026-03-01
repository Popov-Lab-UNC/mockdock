"""
AceGen-PPOD x FCGMB Benchmark Evaluation
==========================================
Runs AceGen's PPOD algorithm against the six FCGMB benchmarks.

PPOD is PPO+Diversity: PPO with the experience replay buffer enabled
(experience_replay=True in config.yaml), which improves diversity and
sample efficiency by replaying high-reward generated molecules.

FCGMB-specific adaptations:
  - Scoring via FCGMBOracle (docking-based, fragment-constrained).
  - Fragment conditioning: the benchmark's required fragment is converted to a
    PromptSMILES scaffold, forcing the model to generate fragment-containing
    molecules throughout training.
  - Initial compounds: the lowest-quartile bioactivity compounds from ChEMBL
    are pre-scored through the oracle before the RL loop begins. This warms up
    the docking infrastructure and establishes baseline docking data.
  - Experience replay: PPOD's prioritised replay buffer accumulates
    high-reward generated molecules during training.
"""

from __future__ import annotations

# ── Algorithm identity (only these three lines change between algorithm folders) ──
ALGORITHM = "ppod"          # identifies this experiment in outputs/logs
SCRIPT_NAME = "ppo"         # subfolder name under acegen-open/scripts/
FUNCTION_NAME = "run_ppo"   # callable inside that script (PPOD reuses PPO's function)

import importlib.util
import logging
import os
import pathlib

import click
import torch
import yaml
from omegaconf import OmegaConf, open_dict

from fcgmb import FCGMBOracle

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ACEGEN_ROOT = SCRIPT_DIR.parents[2] / "acegen-open"
BENCHMARKS = ["AKT1", "CHK1", "ITK", "PCK1", "TTK", "VEGFR2"]


# ── Interface bridge ──────────────────────────────────────────────────────────

class FCGMBTask:
    """Adapts FCGMBOracle to the (smiles: list) -> list[float] interface
    that AceGen algorithm functions expect, with a .finished property."""

    def __init__(self, oracle: FCGMBOracle):
        self.oracle = oracle
        self.name = f"fcgmb_{oracle.benchmark_name}"

    @property
    def finished(self) -> bool:
        return self.oracle.status != "active"

    def __call__(self, smiles: list) -> list:
        scores_dict = self.oracle.score(smiles)
        return [float(scores_dict.get(smi, 0.0)) for smi in smiles]


# ── Fragment conditioning ─────────────────────────────────────────────────────

# ── Initial compound warmup ───────────────────────────────────────────────────

def warmup_oracle(oracle: FCGMBOracle, output_dir: pathlib.Path, n: int = 25) -> None:
    """Pre-score a small set of initial compounds to warm up docking and
    establish baseline data.  Uses at most *n* oracle calls."""
    initial_df = oracle.get_initial_compounds()
    if initial_df.is_empty():
        log.info("No initial compounds available for warmup.")
        return

    smiles_col = "canonical_smiles"
    subset = initial_df.head(n)
    smiles_list = subset[smiles_col].to_list()

    if "score" in subset.columns:
        log.info("Using pre-computed docking scores for %d warmup compounds.", len(smiles_list))
        scores = {row["canonical_smiles"]: row["score"] for row in subset.iter_rows(named=True)}
    else:
        log.info("Warming up oracle with %d initial compounds …", len(smiles_list))
        scores = oracle.score(smiles_list)

    nonzero = sum(1 for v in scores.values() if v > 0)
    log.info("Warmup complete: %d/%d compounds scored > 0.", nonzero, len(smiles_list))

    warmup_csv = output_dir / "initial_compounds_warmup.csv"
    with open(warmup_csv, "w") as fh:
        fh.write("smiles,score\n")
        for smi, score in scores.items():
            fh.write(f"{smi},{score}\n")
    log.info("Warmup results saved to %s", warmup_csv)


# ── Algorithm import ──────────────────────────────────────────────────────────

def _import_algorithm_fn(acegen_root: pathlib.Path):
    """Import the algorithm function from the acegen-open scripts directory."""
    script_path = acegen_root / "scripts" / SCRIPT_NAME / f"{SCRIPT_NAME}.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"AceGen script not found: {script_path}\n"
            f"Make sure acegen-open is at {acegen_root}"
        )
    spec = importlib.util.spec_from_file_location(
        f"_acegen_{ALGORITHM}_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    _saved_cwd = os.getcwd()
    spec.loader.exec_module(module)
    os.chdir(_saved_cwd)
    return getattr(module, FUNCTION_NAME)


# ── Config builder ────────────────────────────────────────────────────────────

def _build_cfg(base_cfg, benchmark: str, seed: int, budget: int, output_dir: pathlib.Path):
    """Merge base config with per-run dynamic values."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = str(output_dir / f"{ALGORITHM}_{benchmark}_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    with open_dict(base_cfg):
        base_cfg.seed = seed
        base_cfg.total_smiles = budget
        base_cfg.log_dir = str(output_dir)
        base_cfg.save_dir = save_dir
        base_cfg.experiment_name = f"fcgmb_{benchmark}"
        base_cfg.agent_name = ALGORITHM
        base_cfg.logger_backend = None

    with open(pathlib.Path(save_dir) / "config.yaml", "w") as fh:
        yaml.dump(OmegaConf.to_container(base_cfg, resolve=True), fh, default_flow_style=False)

    return base_cfg


# ── Single benchmark run ──────────────────────────────────────────────────────

def run_benchmark(
    benchmark: str,
    budget: int,
    seed: int,
    outputs_root: pathlib.Path,
    acegen_root: pathlib.Path,
    n_warmup: int,
):
    log.info("=" * 60)
    log.info(" Benchmark : %s", benchmark)
    log.info(" Algorithm : %s", ALGORITHM)
    log.info(" Budget    : %d", budget)
    log.info(" Seed      : %d", seed)
    log.info("=" * 60)

    output_dir = outputs_root / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle = FCGMBOracle(benchmark, budget=budget)
    log.info("Run directory: %s", oracle.run_dir)

    if n_warmup > 0:
        warmup_oracle(oracle, output_dir, n=n_warmup)

    remaining = oracle.budget_remaining
    log.info("Oracle budget remaining after warmup: %d", remaining)

    config_path = SCRIPT_DIR / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg = _build_cfg(cfg, benchmark, seed, remaining, output_dir)

    # ── Fragment conditioning via oracle.fragment_smiles_with_dummies ────────
    if oracle.fragment_smiles_with_dummies:
        log.info("Fragment SMILES with dummies: %s", oracle.fragment_smiles_with_dummies)
        with open_dict(cfg):
            cfg.promptsmiles = oracle.fragment_smiles_with_dummies
    else:
        log.warning(
            "fragment_smiles_with_dummies not set for %s — running without scaffold conditioning. "
            "Add it to fcgmb/configs/%s.yaml to enable PromptSMILES.",
            benchmark, benchmark,
        )

    run_fn = _import_algorithm_fn(acegen_root)

    from acegen.script_helpers import set_seed
    set_seed(seed)

    task = FCGMBTask(oracle)
    log.info("Starting %s RL loop …", ALGORITHM)
    try:
        run_fn(cfg, task)
    finally:
        try:
            oracle.export_top_poses(n=10)
        except Exception as exc:
            log.warning("Could not export top poses: %s", exc)
        oracle.save_metrics(extra={"model": f"acegen-{ALGORITHM}", "seed": seed})
        log.info(
            "Benchmark %s complete. Budget used: %d/%d, rounds: %d",
            benchmark, oracle.budget_used, oracle.max_budget, oracle.generation_round,
        )
    torch.cuda.empty_cache()


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--benchmark", "benchmarks", multiple=True,
              type=click.Choice(BENCHMARKS, case_sensitive=False),
              help="Benchmark(s) to run. Defaults to all six.")
@click.option("--budget", default=5000, show_default=True,
              help="Maximum oracle (docking) calls per benchmark.")
@click.option("--seed", default=0, show_default=True,
              help="Random seed.")
@click.option("--out", default=None,
              help="Output directory root. Defaults to <script_dir>/outputs.")
@click.option("--acegen-root", default=None,
              help="Path to the acegen-open repository. Auto-detected if not given.")
@click.option("--n-warmup", default=25, show_default=True,
              help="Number of initial compounds to pre-score as oracle warmup (0 to skip).")
def main(benchmarks, budget, seed, out, acegen_root, n_warmup):
    """Run AceGen-PPOD against FCGMB benchmarks."""
    benchmarks = list(benchmarks) if benchmarks else BENCHMARKS
    outputs_root = pathlib.Path(out) if out else SCRIPT_DIR / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    acegen_path = pathlib.Path(acegen_root) if acegen_root else ACEGEN_ROOT

    if not acegen_path.exists():
        raise SystemExit(
            f"acegen-open not found at {acegen_path}. "
            "Pass --acegen-root or ensure it is at the expected location."
        )

    log.info("acegen-open root : %s", acegen_path)
    log.info("Benchmarks       : %s", benchmarks)
    log.info("Output root      : %s", outputs_root)

    for bm in benchmarks:
        run_benchmark(
            benchmark=bm,
            budget=budget,
            seed=seed,
            outputs_root=outputs_root,
            acegen_root=acegen_path,
            n_warmup=n_warmup,
        )

    log.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
