import requests
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Union, Optional, Tuple, List
import numpy as np

class PDBDownloadError(Exception):
    """Raised when a PDB file cannot be downloaded from RCSB."""
    pass

class LigandNotFoundError(Exception):
    """Raised when the specified ligand resname is not found in the structure."""
    pass

class GridPrepError(Exception):
    """Raised when mk_prepare_receptor or AutoGrid fails."""
    pass

try:
    from prody import fetchPDB, parsePDB, writePDB, writePDBStream, confProDy
    confProDy(verbosity='error')
except ImportError:
    pass

def extract_protein_and_ligand(
    pdb_id: str, 
    chain: str = 'A', 
    output_dir: Union[str, Path] = ".",
    ligand_resname: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Fetch a PDB and save cleaned protein (specific chain) and the specified ligand to PDB files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_path = None
    try:
        pdb_path = fetchPDB(pdb_id, folder=str(output_dir), compressed=False)
    except Exception as e:
        print(f"   ProDy fetchPDB failed: {e}")
    
    # fetchPDB can return None instead of raising exception, so check explicitly
    if pdb_path is None:
        print(f"   Falling back to direct RCSB download...")
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            response = requests.get(url)
            response.raise_for_status()
            pdb_path = output_dir / f"{pdb_id.lower()}.pdb"
            with open(pdb_path, "w") as f:
                f.write(response.text)
        except Exception as e:
            raise PDBDownloadError(f"Failed to download PDB {pdb_id}: {e}")
    
    structure = parsePDB(str(pdb_path), altloc='first')
    
    # Auto-detect the correct chain based on ligand location if ligand_resname is provided
    detected_chain = chain
    ligand_res = None
    
    if ligand_resname:
        # First, find which chain(s) contain this ligand
        all_ligands = structure.select(f"resname {ligand_resname}")
        if all_ligands is None:
            raise LigandNotFoundError(f"Ligand with resname {ligand_resname} not found in PDB {pdb_id}")
        
        # Get the FIRST ligand instance and store it
        ligand_res = all_ligands.getHierView().iterResidues().__next__()
        detected_chain = ligand_res.getChid()
        
        if detected_chain != chain:
            print(f"   Note: Ligand '{ligand_resname}' found in chain {detected_chain} (not {chain}). Using chain {detected_chain}.")
    
    # Select protein atoms from the detected chain
    protein_sel = structure.select(f'protein and chain {detected_chain}')
    if protein_sel is None:
        raise ValueError(f"No protein atoms found in chain {detected_chain} for PDB {pdb_id}")
    
    protein_pdb = output_dir / f"{pdb_id}_protein.pdb"
    writePDB(str(protein_pdb), protein_sel)
    
    # Select ligand
    if ligand_resname:
        # Use the SAME first instance we found above
        ligand_sel = structure.select(f"chain {ligand_res.getChid()} and resname {ligand_res.getResname()} and resnum {ligand_res.getResnum()}")
    else:
        # Fallback to first non-protein, non-water residue
        ligands = structure.select('not protein and not water')
        if ligands is None:
            raise ValueError(f"No ligands found in PDB {pdb_id}")
        
        # Get the first residue in the ligand selection
        res = ligands.getHierView().iterResidues().__next__()
        ligand_sel = structure.select(f"chain {res.getChid()} and resname {res.getResname()} and resnum {res.getResnum()}")
    
    ligand_pdb = output_dir / f"{pdb_id}_ligand.pdb"
    writePDB(str(ligand_pdb), ligand_sel)
    
    return protein_pdb, ligand_pdb

class ReceptorPreparer:
    def __init__(self, autogrid_executable: str = "autogrid4", mk_prepare_receptor_executable: str = "mk_prepare_receptor.py"):
        self.autogrid_executable = shutil.which(autogrid_executable)

        # Fallback if shutil.which didn't find it but it's a relative path or in current dir
        if self.autogrid_executable is None:
             if Path(autogrid_executable).exists():
                 self.autogrid_executable = str(Path(autogrid_executable).resolve())
             elif Path.cwd().joinpath(autogrid_executable).exists():
                 self.autogrid_executable = str(Path.cwd().joinpath(autogrid_executable).resolve())
             else:
                 # Last ditch: keep it as is, maybe subprocess finds it?
                 self.autogrid_executable = autogrid_executable
        else:
             # Ensure absolute path even if found by which, if it looks relative
             if not Path(self.autogrid_executable).is_absolute():
                  self.autogrid_executable = str(Path(self.autogrid_executable).resolve())

        self.mk_prepare_receptor_executable = shutil.which(mk_prepare_receptor_executable) or mk_prepare_receptor_executable

    def prepare_receptor_and_grid(
        self, 
        pdb_id: str, 
        chain: str = 'A', 
        output_dir: Union[str, Path] = ".", 
        allow_bad_res: bool = False,
        ligand_resname: Optional[str] = None,
        protein_pdb_path: Optional[Union[str, Path]] = None,
        ligand_pdb_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Prepare receptor PDBQT and GPF using mk_prepare_receptor.py and run AutoGrid4.
        Returns the path to the .maps.fld file.

        If protein_pdb_path and ligand_pdb_path are provided, they are used instead of fetching from PDB.
        """
        output_dir = Path(output_dir)
        grid_dir = output_dir / "grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Obtain protein and ligand PDBs
        if protein_pdb_path and ligand_pdb_path:
            protein_pdb_path = Path(protein_pdb_path)
            ligand_pdb_path = Path(ligand_pdb_path)

            if not protein_pdb_path.exists():
                raise FileNotFoundError(f"Protein PDB not found: {protein_pdb_path}")
            if not ligand_pdb_path.exists():
                raise FileNotFoundError(f"Ligand PDB not found: {ligand_pdb_path}")

            # Copy to grid dir for consistency
            protein_pdb = grid_dir / protein_pdb_path.name
            ligand_pdb = grid_dir / ligand_pdb_path.name
            shutil.copy2(protein_pdb_path, protein_pdb)
            shutil.copy2(ligand_pdb_path, ligand_pdb)

            print(f"Using provided protein ({protein_pdb.name}) and ligand ({ligand_pdb.name})")

        else:
            protein_pdb, ligand_pdb = extract_protein_and_ligand(pdb_id, chain=chain, output_dir=grid_dir, ligand_resname=ligand_resname)
        
        # 2. Run mk_prepare_receptor.py
        base_name = f"rec_{pdb_id.lower()}"
        cmd = [
            self.mk_prepare_receptor_executable,
            "--read_pdb", str(protein_pdb.name),
            "-o", base_name,
            "-p", "-g",
            "--box_enveloping", str(ligand_pdb.name),
            "--padding", "5"
        ]
        
        if allow_bad_res:
            cmd.append("--allow_bad_res")
            
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(grid_dir))
        
        if result.returncode != 0:
            raise GridPrepError(f"Receptor preparation failed (mk_prepare_receptor.py returned {result.returncode}).")
        
        gpf_path = grid_dir / f"{base_name}.gpf"
        glg_path = grid_dir / f"{base_name}.glg"
        
        # 3. Run AutoGrid4
        print(f"Running AutoGrid4 for {gpf_path.name} using {self.autogrid_executable}...")
        
        # Check grid size to warn user
        try:
            with open(gpf_path, 'r') as f:
                for line in f:
                    if line.startswith('npts'):
                        print(f"Grid size: {line.strip()}")
                        break
        except:
            pass

        ag_cmd = [self.autogrid_executable, "-p", gpf_path.name, "-l", glg_path.name]
        # Run without capture_output to show progress in real-time
        ag_result = subprocess.run(ag_cmd, cwd=str(grid_dir))
        
        if ag_result.returncode != 0:
            raise GridPrepError(f"AutoGrid4 failed (returned {ag_result.returncode}). Check {glg_path} for details.")
            
        fld_path = grid_dir / f"{base_name}.maps.fld"
        return fld_path

    # Keep compatibility with previous API if needed, but redirects to the new one
    def prepare_receptor(self, pdb_id: str, chain: str = 'A', output_dir: Union[str, Path] = ".", allow_bad_res: bool = False) -> Path:
        # For compatibility, returns the pdbqt path
        grid_dir = Path(output_dir) / "grid"
        if not (grid_dir / f"{pdb_id}_receptor.pdbqt").exists():
            self.prepare_receptor_and_grid(pdb_id, chain, output_dir, allow_bad_res)
        return grid_dir / f"{pdb_id}_receptor.pdbqt"

    def generate_grid(self, receptor_pdbqt: Union[str, Path], *args, **kwargs) -> Path:
        # For compatibility, returns the fld path
        # If it was already generated by prepare_receptor_and_grid, just return it
        fld_path = Path(receptor_pdbqt).with_suffix(".maps.fld")
        if fld_path.exists():
            return fld_path
        # Otherwise this indicates the old API flow was used, which we want to discourage
        raise NotImplementedError("Old generate_grid API is deprecated. Use prepare_receptor_and_grid.")
