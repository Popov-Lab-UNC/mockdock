from pathlib import Path

from vina import Vina


def test_vina_ad4_maps_can_be_loaded_and_docked():
    fixture_dir = Path(__file__).parent / "2R0U"
    map_prefix = fixture_dir / "rec_2r0u"
    ligand_path = fixture_dir / "lig_0_s0.pdbqt"

    v = Vina(sf_name="ad4", cpu=0)
    v.load_maps(str(map_prefix))
    v.set_ligand_from_file(str(ligand_path))

    energy = v.score()
    assert len(energy) > 0

    v.dock(exhaustiveness=32, n_poses=10)
