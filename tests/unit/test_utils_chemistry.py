from __future__ import annotations

from rdkit import Chem

from mockdock.utils import check_2d_match, get_robust_match, standardize_smiles


def test_standardize_smiles_invalid_returns_none():
    assert standardize_smiles("not-a-smiles") is None


def test_standardize_smiles_removes_counterion():
    out = standardize_smiles("CC(=O)[O-].[Na+]")
    assert out == "CC(=O)O"


def test_check_2d_match_true_for_fragment():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    frag = Chem.MolFromSmiles("c1ccccc1")
    assert check_2d_match(mol, frag) is True


def test_get_robust_match_returns_empty_for_nonmatch():
    mol = Chem.MolFromSmiles("CCO")
    frag = Chem.MolFromSmiles("c1ccccc1")
    assert get_robust_match(mol, frag) == ()
