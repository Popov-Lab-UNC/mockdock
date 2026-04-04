from .analysis import DockingAnalyzer
from .data import fetch_chembl_data
from .docking import AutoDockGPUOracle, AutoDockVinaOracle, DockingOracle
from .evaluator import FCGMBEvaluator
from .ligand_prep import LigandPreparer
from .loader import BenchmarkLoader
from .oracle import FCGMBOracle
from .utils import (
    assign_bond_orders_from_template,
    check_2d_match,
    detect_gpus,
    fetch_ligand_expo_sdf,
    plot_activity_distribution,
    plot_docking_results,
    resolve_backend,
    standardize_smiles,
)


def __getattr__(name):
    if name in ("ReceptorPreparer", "extract_protein_and_ligand"):
        from .receptor import ReceptorPreparer, extract_protein_and_ligand

        globals()["ReceptorPreparer"] = ReceptorPreparer
        globals()["extract_protein_and_ligand"] = extract_protein_and_ligand
        return globals()[name]
    raise AttributeError(f"module 'fcgmb' has no attribute {name!r}")


__all__ = [
    "fetch_chembl_data",
    "AutoDockGPUOracle",
    "AutoDockVinaOracle",
    "DockingOracle",
    "ReceptorPreparer",
    "extract_protein_and_ligand",
    "LigandPreparer",
    "DockingAnalyzer",
    "plot_docking_results",
    "plot_activity_distribution",
    "fetch_ligand_expo_sdf",
    "assign_bond_orders_from_template",
    "FCGMBOracle",
    "FCGMBEvaluator",
    "BenchmarkLoader",
    "detect_gpus",
    "resolve_backend",
    "standardize_smiles",
    "check_2d_match",
]
