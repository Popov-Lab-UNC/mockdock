from __future__ import annotations

from types import SimpleNamespace

from mockdock.oracle import MDOracle


def _build_oracle(loader_cfg):
    oracle = MDOracle.__new__(MDOracle)
    oracle._loader = loader_cfg
    oracle._total_analysis_time = 0.0
    return oracle


def test_analyze_results_normalizes_and_sets_failed_rmsd():
    loader_cfg = SimpleNamespace(require_pose_rmsd=True, low_score=-8.0, high_score=-12.0)
    oracle = _build_oracle(loader_cfg)
    oracle._filter_poses_for_molecule = lambda *_: (-10.0, True, -10.0, "x.dlg", 0)

    scores, batch = oracle._analyze_results([("CC", "CC")], [[{"dlg_path": "x.dlg"}]])
    assert scores["CC"] == 0.5
    assert batch[0]["skip_reason"] is None
    assert batch[0]["valid_pose_found"] is True

    oracle._filter_poses_for_molecule = lambda *_: (float("nan"), False, -10.0, None, -1)
    scores2, batch2 = oracle._analyze_results([("CC", "CC")], [[{"dlg_path": "x.dlg"}]])
    assert scores2["CC"] == -1.5
    assert batch2[0]["skip_reason"] == "failed_rmsd"


def test_analyze_results_zero_denom_branch():
    loader_cfg = SimpleNamespace(require_pose_rmsd=True, low_score=-10.0, high_score=-10.0)
    oracle = _build_oracle(loader_cfg)
    oracle._filter_poses_for_molecule = lambda *_: (-10.0, True, -10.0, "x.dlg", 0)
    scores, _ = oracle._analyze_results([("CC", "CC")], [[{"dlg_path": "x.dlg"}]])
    assert scores["CC"] == 1.0
