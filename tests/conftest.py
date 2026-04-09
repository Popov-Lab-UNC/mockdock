from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def loader_defaults() -> SimpleNamespace:
    return SimpleNamespace(
        require_pose_rmsd=True,
        low_score=-8.0,
        high_score=-12.0,
        fragment_smiles="c1ccccc1",
        rmsd_threshold=2.0,
        require_fragment_match=True,
    )
