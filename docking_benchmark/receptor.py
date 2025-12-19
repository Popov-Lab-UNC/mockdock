import requests
from pathlib import Path
import os
import shutil
import subprocess
from typing import Union, Optional, Tuple, List
import numpy as np

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
    
    try:
        pdb_path = fetchPDB(pdb_id, folder=str(output_dir), compressed=False)
    except Exception:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url)
        response.raise_for_status()
        pdb_path = output_dir / f"{pdb_id.lower()}.pdb"
        with open(pdb_path, "w") as f:
            f.write(response.text)
    
    structure = parsePDB(str(pdb_path), altloc='first')
    
    # Select protein atoms
    protein_sel = structure.select(f'protein and chain {chain}')
    if protein_sel is None:
        raise ValueError(f"No protein atoms found in chain {chain} for PDB {pdb_id}")
    
    protein_pdb = output_dir / f"{pdb_id}_protein.pdb"
    writePDB(str(protein_pdb), protein_sel)
    
    # Select ligand
    if ligand_resname:
        ligand_sel = structure.select(f"resname {ligand_resname}")
        if ligand_sel is None:
             raise ValueError(f"Ligand with resname {ligand_resname} not found in PDB {pdb_id}")
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
        self.autogrid_executable = shutil.which(autogrid_executable) or autogrid_executable
        self.mk_prepare_receptor_executable = shutil.which(mk_prepare_receptor_executable) or mk_prepare_receptor_executable

    def prepare_receptor_and_grid(
        self, 
        pdb_id: str, 
        chain: str = 'A', 
        output_dir: Union[str, Path] = ".", 
        allow_bad_res: bool = False,
        ligand_resname: Optional[str] = None
    ) -> Path:
        """
        Prepare receptor PDBQT and GPF using mk_prepare_receptor.py and run AutoGrid4.
        Returns the path to the .maps.fld file.
        """
        output_dir = Path(output_dir)
        grid_dir = output_dir / "grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Extract protein and ligand PDBs
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
        result = subprocess.run(cmd, cwd=str(grid_dir), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in mk_prepare_receptor.py:\n{result.stderr}")
            # Check if it failed due to template matching
            if "Template matching failed" in result.stderr:
                raise RuntimeError(f"Receptor preparation failed: Template matching failed. Consider using allow_bad_res=True.\nDetails: {result.stderr}")
            raise RuntimeError(f"Receptor preparation failed: {result.stderr}")
        
        gpf_path = grid_dir / f"{base_name}.gpf"
        glg_path = grid_dir / f"{base_name}.glg"
        
        # 3. Run AutoGrid4
        print(f"Running AutoGrid4 for {gpf_path.name}...")
        ag_cmd = [self.autogrid_executable, "-p", gpf_path.name, "-l", glg_path.name]
        ag_result = subprocess.run(ag_cmd, cwd=str(grid_dir), capture_output=True, text=True)
        
        if ag_result.returncode != 0:
            raise RuntimeError(f"AutoGrid4 failed: {ag_result.stderr}")
            
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
