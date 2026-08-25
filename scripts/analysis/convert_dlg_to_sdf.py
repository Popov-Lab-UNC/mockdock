#!/usr/bin/env python3
# scripts/analysis/convert_dlg_to_sdf.py
"""
Utility script to convert AutoDock/Gnina .dlg docking output files into standard RDKit .sdf files.
Allows extracting all conformers/poses or a specific conformation index.

Usage:
    python3 scripts/analysis/convert_dlg_to_sdf.py -i path/to/pose.dlg -o output.sdf [--pose-index 3] [--smiles "CO..."]
"""

import argparse
import sys
from pathlib import Path

# Add src folder to sys.path so we can import helper modules if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure you are running in a virtual environment where 'meeko' and 'rdkit' are installed.")
    sys.exit(1)


def unroll_multiconf_rdkit_mols(rdkit_mols: list[Chem.Mol]) -> list[Chem.Mol]:
    """Split one RDKit mol with multiple conformers into one mol per conformer."""
    if len(rdkit_mols) == 1 and rdkit_mols[0].GetNumConformers() > 1:
        base_mol = rdkit_mols[0]
        unrolled: list[Chem.Mol] = []
        for conf in base_mol.GetConformers():
            new_mol = Chem.Mol(base_mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(conf, assignId=True)
            unrolled.append(new_mol)
        return unrolled
    return rdkit_mols


def convert_dlg_to_sdf(dlg_path: Path, sdf_path: Path, pose_index: int = None, smiles: str = None):
    if not dlg_path.exists():
        raise FileNotFoundError(f"Input DLG file not found at: {dlg_path}")

    # Read PDBQT/DLG file
    is_dlg = dlg_path.suffix.lower() == ".dlg"
    try:
        pdbqt_mol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=is_dlg, skip_typing=True)
    except Exception as e:
        print(f"Error parsing {dlg_path} using meeko: {e}")
        sys.exit(1)

    # Convert to RDKit molecules
    try:
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    except Exception as e:
        print(f"Error generating RDKit molecules: {e}")
        sys.exit(1)

    if not rdkit_mols:
        print(f"No RDKit molecules created from {dlg_path}")
        sys.exit(1)

    rdkit_mols = unroll_multiconf_rdkit_mols(rdkit_mols)

    # Collect free energies from meeko pose data if available
    energies = []
    if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
        energies = list(pdbqt_mol._pose_data["free_energies"])

    # Prepare writer
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(sdf_path))

    if pose_index is not None:
        if pose_index < 0 or pose_index >= len(rdkit_mols):
            print(f"Error: Requested pose index {pose_index} is out of bounds. File has {len(rdkit_mols)} poses.")
            writer.close()
            sys.exit(1)
        mols_to_write = [(pose_index, rdkit_mols[pose_index])]
    else:
        mols_to_write = list(enumerate(rdkit_mols))

    for idx, mol in mols_to_write:
        # Set basic properties
        mol.SetProp("_Name", f"{dlg_path.stem}_pose_{idx}")
        mol.SetProp("pose_index", str(idx))
        mol.SetProp("source_dlg", str(dlg_path.resolve()))
        if smiles:
            mol.SetProp("SMILES", smiles)
        if idx < len(energies):
            mol.SetProp("free_energy", f"{energies[idx]:.2f}")
        writer.write(mol)

    writer.close()
    print(f"Successfully wrote {len(mols_to_write)} poses to {sdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert AutoDock .dlg files to RDKit-compatible .sdf files.")
    parser.add_argument("-i", "--input", required=True, help="Path to input .dlg file")
    parser.add_argument("-o", "--output", help="Path to output .sdf file (default: input name with .sdf extension)")
    parser.add_argument("-p", "--pose-index", type=int, default=None, help="Specific pose index to extract (0-based)")
    parser.add_argument("--smiles", help="SMILES string to attach as property to the SDF molecule(s)")
    
    args = parser.parse_args()
    
    dlg_path = Path(args.input)
    if args.output:
        sdf_path = Path(args.output)
    else:
        sdf_path = dlg_path.with_suffix(".sdf")
        
    convert_dlg_to_sdf(dlg_path, sdf_path, args.pose_index, args.smiles)


if __name__ == "__main__":
    main()
