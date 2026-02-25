# Pre-Built Grids Directory

This directory contains AutoGrid 4 maps pre-computed for each benchmark PDB structure.

## Directory Structure

```
grids/
└── <PDB_ID>/              # One sub-directory per PDB
    ├── <PDB_ID>.maps.fld  # AutoGrid field file (entry point for docking)
    ├── <PDB_ID>.A.map     # Per-atom-type grid maps
    ├── <PDB_ID>.C.map
    ├── ...
    ├── <PDB_ID>_ligand_corrected.sdf   # Reference ligand (for RMSD scoring)
    └── <PDB_ID>_protein_fixed.pdbqt   # Prepared receptor
```

## Adding Your Own Grids

To add grids for a custom benchmark:

1. Create `fcgmb/grids/<YOUR_PDB_ID>/` and place the AutoGrid map files inside.
2. Include the reference ligand as `<PDB_ID>_ligand_corrected.sdf` (with correct bond orders).
3. The oracle will automatically find the grid at docking time.

Alternatively, you can let the oracle auto-prepare grids (requires `autogrid4`, `mk_prepare_receptor.py`, and `mmtbx.reduce2`). They will be saved under `.fcgmb/grids/<PDB_ID>/` in the current working directory.

## Current Benchmarks

| Benchmark | PDB ID | Target       |
|-----------|--------|--------------|
| AKT1      | 4EJN   | CHEMBL4282   |
| CHK1      | 2R0U   | CHEMBL4630   |
| ITK       | 3QGW   | CHEMBL2959   |
| PCK1      | 1NHX   | CHEMBL2911   |
| TTK       | 3WZJ   | CHEMBL3983   |
| VEGFR2    | 3VHE   | CHEMBL279    |
