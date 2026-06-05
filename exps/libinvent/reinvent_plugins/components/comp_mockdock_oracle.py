"""REINVENT4 scoring component backed by mockdock MDOracle."""

from __future__ import annotations

__all__ = ["MockdockOracle"]

import atexit
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from mockdock import MDOracle

from .add_tag import add_tag
from .component_results import ComponentResults

logger = logging.getLogger("reinvent")


class BudgetExhaustedStop(RuntimeError):
    """Signal REINVENT to stop once scorer budget is consumed."""


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for MockdockOracle component endpoints."""

    benchmark: List[str]
    budget: List[int]
    run_dir: List[str]
    docking_backend: List[str]
    clip_reward_upper_bound: Optional[List[bool]] = None


@add_tag("__component")
class MockdockOracle:
    """Score molecules with mockdock and return normalized rewards [0, 1]."""

    def __init__(self, params: Parameters):
        self._benchmark = params.benchmark[0]
        self._budget = int(params.budget[0])
        self._run_dir = Path(params.run_dir[0]).resolve()
        self._backend = params.docking_backend[0]
        self._clip_reward_upper_bound = (
            bool(params.clip_reward_upper_bound[0])
            if params.clip_reward_upper_bound is not None
            else True
        )
        self._finalized = False
        self._generation_time_sec = 0.0
        self._generated_ligands = 0
        self._consumed_budget = 0
        self._last_call_end_ts = None
        self._budget_stop_marker = self._run_dir / "budget_exhausted.json"

        clip_values = (
            params.clip_reward_upper_bound
            if params.clip_reward_upper_bound is not None
            else [self._clip_reward_upper_bound] * len(params.benchmark)
        )
        for benchmark, budget, run_dir, backend, clip_reward_upper_bound in zip(
            params.benchmark,
            params.budget,
            params.run_dir,
            params.docking_backend,
            clip_values,
        ):
            if (
                benchmark != self._benchmark
                or int(budget) != self._budget
                or Path(run_dir).resolve() != self._run_dir
                or backend != self._backend
                or bool(clip_reward_upper_bound) != self._clip_reward_upper_bound
            ):
                raise ValueError(
                    "MockdockOracle supports only one benchmark/run_dir/backend/clip mode per component."
                )

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._oracle = MDOracle(
            benchmark_name=self._benchmark,
            budget=self._budget,
            run_dir=self._run_dir,
            docking_backend=self._backend,
            clip_reward_upper_bound=self._clip_reward_upper_bound,
        )
        atexit.register(self._finalize)
        logger.info(
            "MockdockOracle initialized: benchmark=%s budget=%d run_dir=%s backend=%s clip_reward_upper_bound=%s",
            self._benchmark,
            self._budget,
            self._run_dir,
            self._backend,
            self._clip_reward_upper_bound,
        )

    def __call__(self, smiles: List[str]) -> ComponentResults:
        now = time.perf_counter()
        if self._last_call_end_ts is not None:
            # Time between scorer calls is dominated by model generation/training.
            self._generation_time_sec += max(0.0, now - self._last_call_end_ts)
        n_generated = len(smiles)
        self._generated_ligands += n_generated

        # Account every generated SMILES against budget, including duplicates.
        remaining_budget = max(0, self._budget - self._consumed_budget)
        n_to_consume = min(n_generated, remaining_budget)
        self._consumed_budget += n_to_consume

        if n_to_consume > 0:
            score_map = self._oracle.score(smiles[:n_to_consume])
        else:
            score_map = {}
        self._last_call_end_ts = time.perf_counter()
        ordered_scores = np.zeros(n_generated, dtype=float)
        for idx, smi in enumerate(smiles[:n_to_consume]):
            ordered_scores[idx] = self._to_rl_reward(float(score_map.get(smi, 0.0)))
        if self._consumed_budget >= self._budget:
            self._finalize()
            self._budget_stop_marker.write_text(
                json.dumps(
                    {
                        "benchmark": self._benchmark,
                        "budget_total": self._budget,
                        "budget_used": self._consumed_budget,
                        "timestamp_unix": int(time.time()),
                    },
                    indent=2,
                    sort_keys=False,
                )
                + "\n",
                encoding="utf-8",
            )
            raise BudgetExhaustedStop(
                f"MOCKDOCK_BUDGET_EXHAUSTED benchmark={self._benchmark} budget={self._budget}"
            )
        return ComponentResults([ordered_scores])

    def _to_rl_reward(self, raw_score: float) -> float:
        """Map oracle scores to stable RL rewards in [0, 1]."""
        non_negative = max(0.0, raw_score)
        if self._clip_reward_upper_bound:
            return min(1.0, non_negative)
        # With unclipped oracle scores (>1 possible), DAP updates can become too
        # aggressive and collapse to near-duplicate batches. This smooth cap
        # preserves ranking while keeping reward magnitude bounded.
        return non_negative / (1.0 + non_negative)

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        try:
            self._oracle.export_top_poses(n=10)
        except Exception as exc:  # pragma: no cover
            logger.warning("MockdockOracle export_top_poses failed: %s", exc)
        try:
            self._oracle.save_metrics(
                extra={
                    "model": "reinvent4-libinvent",
                    "benchmark": self._benchmark,
                    "total_generation_time_sec": self._generation_time_sec,
                    "n_generated_ligands": self._generated_ligands,
                    "consumed_budget_generated_smiles": self._consumed_budget,
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("MockdockOracle save_metrics failed: %s", exc)
        self._rewrite_budget_accounting()

    def _rewrite_budget_accounting(self):
        consumed_budget = min(self._consumed_budget, self._budget)
        for relpath in ("status.json", "metrics.json"):
            path = self._run_dir / relpath
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["budget_total"] = int(data.get("budget_total", self._budget))
                data["budget_used"] = consumed_budget
                data["n_molecules_total"] = consumed_budget
                path.write_text(
                    json.dumps(data, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("MockdockOracle failed to update %s: %s", path, exc)

