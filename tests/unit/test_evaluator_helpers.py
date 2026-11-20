from __future__ import annotations

import polars as pl

from mockdock.evaluator import MDEvaluator


def test_auc_top_k_expected_value():
    df = pl.DataFrame({"reward_score": [0.1, 0.4, 0.2, 0.8]})
    auc = MDEvaluator._auc_top_k(df, k=2)
    assert 0.0 <= auc <= 1.0
    # Running top-2 (zero-padded at step 1): [0.05, 0.25, 0.30, 0.60]
    # Trapezoid area on x=1..4: 0.5*(0.05+0.25) + 0.5*(0.25+0.30) + 0.5*(0.30+0.60) = 0.875
    # Normalized by (N - 1) = 3: 0.875 / 3 = 0.2917
    assert round(auc, 4) == 0.2917


def test_novelty_with_empty_reference_is_all_novel():
    novelty = MDEvaluator._novelty(["C", "CC"], set())
    assert novelty == 1.0
