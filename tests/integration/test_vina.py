from __future__ import annotations

from pathlib import Path

import pytest

vina = pytest.importorskip("vina")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_vina
def test_vina_ad4_maps_can_be_loaded_and_docked():
    repo_root = Path(__file__).parents[2]
    map_prefix = repo_root / "src" / "mockdock" / "grids" / "2R0U" / "rec_2r0u"
    ligand_path = Path(__file__).parents[1] / "fixtures" / "lig_0_s0.pdbqt"

    if not (map_prefix.parent / f"{map_prefix.name}.maps.fld").exists():
        pytest.skip(f"Receptor map not found at {map_prefix}")

    v = vina.Vina(sf_name="ad4", cpu=1)
    v.load_maps(str(map_prefix))
    v.set_ligand_from_file(str(ligand_path))
    energy = v.score()
    assert len(energy) > 0
    v.dock(exhaustiveness=1, n_poses=2)
