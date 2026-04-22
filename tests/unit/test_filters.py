# tests/unit/test_filters.py
from __future__ import annotations

from rdkit import Chem
from mockdock.filters import MDFilters

def test_filters_pains_hit():
    # p-Benzoquinone (PAINS hit)
    smiles = "O=C1C=CC(=O)C=C1"
    mol = Chem.MolFromSmiles(smiles)
    filters = MDFilters(active_rulesets=["PAINS"])
    res = filters.evaluate(mol)
    assert res["pass"] is False
    assert "PAINS" in res["rules_hit"]

def test_filters_bms_hit():
    filters = MDFilters(active_rulesets=["BMS"])
    mol_ok = Chem.MolFromSmiles("c1ccccc1CCCCC") # MW > 100
    assert filters.passes(mol_ok) is True

def test_rules_loaded():
    filters = MDFilters(active_rulesets=["PAINS", "BMS"])
    assert len(filters._catalog) > 0

def test_filters_physchem_bounds():
    filters = MDFilters()
    
    # Too small
    mol_small = Chem.MolFromSmiles("C")
    assert filters.evaluate(mol_small)["pass"] is False
    
    # Too large (fake large molecule)
    mol_large = Chem.MolFromSmiles("C" * 100)
    assert filters.evaluate(mol_large)["pass"] is False
    
    # OK
    mol_ok = Chem.MolFromSmiles("c1ccccc1CCC")
    assert filters.evaluate(mol_ok)["pass"] is True

def test_filters_logp():
    filters = MDFilters()
    # High LogP
    mol_greasy = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")
    assert filters.evaluate(mol_greasy)["pass"] is False
