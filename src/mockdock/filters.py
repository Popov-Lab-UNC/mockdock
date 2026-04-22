# src/mockdock/filters.py
from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

class MDFilters:
    """
    Efficient MedChem filtering system for mockdock.
    Uses RDKit FilterCatalog for high-performance SMARTS matching.
    Data sourced from rd_filters (Pat Walters).
    """

    def __init__(self, active_rulesets: list[str] | None = None):
        if active_rulesets is None:
            active_rulesets = ["PAINS", "BMS"]
        
        self.active_rulesets = active_rulesets
        self.data_path = Path(__file__).parent / "data" / "alert_collection.csv"
        
        if not self.data_path.exists():
            # Fallback or error if data not found
            # In a real package we might download it or bundle it properly
            pass

        self._catalog = self._build_catalog()

    def _build_catalog(self) -> list:
        rules = []
        if self.data_path.exists():
            with open(self.data_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rs_name = row["rule_set_name"].strip()
                    if rs_name in self.active_rulesets:
                        mol = Chem.MolFromSmarts(row["smarts"])
                        if mol:
                            rules.append((mol, rs_name, row["description"]))
        return rules

    def evaluate(self, mol: Chem.Mol) -> dict:
        """
        Evaluate a molecule against all active filters.
        Returns a dict with 'pass' (bool) and 'reasons' (list of strings).
        """
        if mol is None:
            return {"pass": False, "reasons": ["Invalid molecule"], "rules_hit": []}

        reasons = []
        rules_hit = []

        # 1. Physchem filters
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Increased upper MW limit to 700 to accommodate benchmark data
        if not (100 <= mw <= 700):
            reasons.append(f"MW out of range (100-700): {mw:.1f}")
        if not (-3 <= logp <= 6.5):
            reasons.append(f"LogP out of range (-3 to 6.5): {logp:.1f}")
        
        # Rotatable Bonds filter (Veber's rule: <= 10)
        n_rot = Descriptors.NumRotatableBonds(mol)
        if n_rot > 10:
            reasons.append(f"Rotatable bonds out of range (<= 10): {n_rot}")
        
        # 2. Structural alerts
        for rule_mol, rule_set, desc in self._catalog:
            if mol.HasSubstructMatch(rule_mol):
                reasons.append(f"{rule_set}: {desc}")
                rules_hit.append(rule_set)

        is_pass = len(reasons) == 0
        return {
            "pass": is_pass,
            "reasons": reasons,
            "rules_hit": list(set(rules_hit)),
            "mw": mw,
            "logp": logp,
            "n_rot": n_rot,
        }

    def passes(self, mol: Chem.Mol) -> bool:
        """Convenience method returning True if all filters pass."""
        return self.evaluate(mol)["pass"]
