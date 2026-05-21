from __future__ import annotations

import polars as pl

from mockdock.evaluator import MDEvaluator


def test_auc_top_k_expected_value():
    df = pl.DataFrame({"reward_score": [0.1, 0.4, 0.2, 0.8]})
    auc = MDEvaluator._auc_top_k(df, k=2)
    assert 0.0 <= auc <= 1.0
    assert round(auc, 4) == 0.225


def test_oracle_efficiency_reaches_target_early():
    df = pl.DataFrame({"reward_score": [0.1, 0.9, 0.2, 0.3]})
    calls = MDEvaluator._oracle_efficiency(df, k=2, frac=0.8)
    assert calls == 2


def test_oracle_efficiency_empty_returns_zero():
    df = pl.DataFrame({"reward_score": []})
    assert MDEvaluator._oracle_efficiency(df, k=10, frac=0.8) == 0


def test_novelty_with_empty_reference_is_all_novel():
    novelty = MDEvaluator._novelty(["C", "CC"], set())
    assert novelty == 1.0
