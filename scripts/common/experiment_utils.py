from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

MODEL_RENAME_MAP = {
    "acegen-a2c": "A2C",
    "acegen-ahc": "AHC",
    "acegen-ppo": "PPO",
    "acegen-ppod": "PPOD",
    "acegen-reinforce": "REINFORCE",
    "acegen-reinvent": "REINVENT",
    "genmol": "GenMol",
    "libinvent": "LibINVENT",
    "invirtuogen": "InVirtuoGen",
}

REFERENCE_SET_LABEL = "Reference Set"
REFERENCE_SET_CACHE_DIRNAME = "reference_set_scores"
REFERENCE_SET_CACHE_FILENAME = "molecule_metrics_cache.csv"
EXPENSIVE_SCORE_COLUMNS = {
    "molskill_score",
    "stoplight_score",
    "aizynthfinder_score",
    "aizynthfinder_state_score",
}
MODEL_PLOT_ORDER = [
    REFERENCE_SET_LABEL,
    "A2C",
    "AHC",
    "PPO",
    "PPOD",
    "REINFORCE",
    "REINVENT",
    "LibINVENT",
    "GenMol",
    "InVirtuoGen",
]


def ensure_src_on_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def effective_yield_filter(df):
    """Return rows in the Effective Yield Rate compound set."""
    import polars as pl

    return df.filter(
        pl.col("is_novel").cast(pl.Boolean) & pl.col("has_fragment").cast(pl.Boolean)
    )
