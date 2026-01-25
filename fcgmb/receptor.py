import requests
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Union, Optional, Tuple
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
    from prody import fetchPDB, parsePDB, parseMMCIF, writePDB, writePDBStream, confProDy
    confProDy(verbosity='error')
except ImportError:
    pass

def _download_mmcif(pdb_id: str, output_dir: Path) -> Path:
    """
    Download an mmCIF for a PDB id into output_dir and return the local path.
    Prefer ProDy's fetchPDB; fall back to direct RCSB download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_path = None
    try:
        # ProDy returns the downloaded file path or None
        structure_path = fetchPDB(pdb_id, folder=str(output_dir), compressed=False, format='cif')
    except Exception as e:
        print(f"   ProDy fetchPDB (cif) failed: {e}")

    if structure_path is None:
        print("   Falling back to direct RCSB download (mmCIF)...")
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        try:
            response = requests.get(url)
            response.raise_for_status()
            structure_path = output_dir / f"{pdb_id.lower()}.cif"
            structure_path.write_text(response.text)
        except Exception as e:
            raise PDBDownloadError(f"Failed to download PDB structure {pdb_id}: {e}")

    return Path(structure_path)


def _parse_mmcif(structure_path: Path):
    """
    Parse mmCIF with ProDy and keep all altlocs.
    """
    try:
        return parseMMCIF(str(structure_path), altloc="all")
    except Exception as e:
        # Provide a higher-signal error message for common ProDy failures.
        raise ValueError(
            f"Failed to parse mmCIF with ProDy: {structure_path} ({type(e).__name__}: {e}). "
            "If this is the ProDy+NumPy issue, prefer using a compatible NumPy/ProDy stack "
            "or switch to PDB download/parse."
        )


def _choose_altloc_token(sel) -> Optional[str]:
    """
    Decide which altloc to use for a selection.

    Returns:
    - '_' to represent blank altloc in ProDy selections
    - 'A'/'B'/... for letter altlocs
    - None if selection has no altloc info or decision fails
    """
    try:
        altlocs = [str(a) for a in sel.getAltlocs()]
        if not altlocs:
            return None
        # Prefer blank altloc if present
        if " " in altlocs:
            return "_"
        from collections import Counter
        return Counter(altlocs).most_common(1)[0][0]
    except Exception:
        return None


def _pick_ligand_instance(structure, ligand_resname: str) -> Tuple[str, str, int, Optional[str]]:
    """
    Find the first instance of a ligand resname and return (chain_id, resname, resnum, altloc_token).
    """
    all_ligands = structure.select(f"resname {ligand_resname}")
    if all_ligands is None:
        raise LigandNotFoundError(f"Ligand with resname {ligand_resname} not found in structure")

    altloc_token = _choose_altloc_token(all_ligands)
    ligands_for_res = all_ligands.select(f"altloc {altloc_token}") if altloc_token else all_ligands
    ligand_res = ligands_for_res.getHierView().iterResidues().__next__()
    return ligand_res.getChid(), ligand_res.getResname(), ligand_res.getResnum(), altloc_token


def _pick_default_ligand(structure) -> Tuple[str, str, int, Optional[str]]:
    """
    Fallback ligand choice when ligand_resname isn't provided.
    Picks the first non-protein, non-water residue.
    """
    ligands = structure.select("not protein and not water")
    if ligands is None:
        raise ValueError("No ligands found (not protein and not water)")

    altloc_token = _choose_altloc_token(ligands)
    ligands_for_res = ligands.select(f"altloc {altloc_token}") if altloc_token else ligands
    res = ligands_for_res.getHierView().iterResidues().__next__()
    return res.getChid(), res.getResname(), res.getResnum(), altloc_token


def _select_protein(structure, chain_id: str):
    """
    Select protein atoms for a chain. Prefer blank altloc (to avoid duplicates), but fall back
    to any altloc if blank selection is empty.
    """
    sel = structure.select(f"protein and chain {chain_id} and altloc _")
    if sel is None:
        sel = structure.select(f"protein and chain {chain_id}")
    if sel is None:
        raise ValueError(f"No protein atoms found in chain {chain_id}")
    return sel


def _select_ligand(structure, chain_id: str, resname: str, resnum: int, altloc_token: Optional[str]):
    """
    Select ligand atoms for a specific residue instance.
    """
    if altloc_token:
        sel = structure.select(f"chain {chain_id} and resname {resname} and resnum {resnum} and altloc {altloc_token}")
        if sel is not None:
            return sel
    sel = structure.select(f"chain {chain_id} and resname {resname} and resnum {resnum}")
    if sel is None:
        raise ValueError(f"Failed to select ligand {resname} {chain_id} {resnum}")
    return sel

def _pick_receptor_chain_from_ligand(structure, ligand_sel) -> str:
    """
    Decide which *protein* chain to dock against using geometric proximity.

    mmCIF/PDB chain naming is inconsistent. We identify the protein chain
    whose atoms are closest (minimum distance) to the ligand atoms.
    """
    # Protein atoms (prefer blank altloc to avoid duplicates)
    prot_all = structure.select("protein and altloc _")
    if prot_all is None:
        prot_all = structure.select("protein")
    if prot_all is None:
        raise ValueError("No protein atoms found in structure")

    protein_chains = sorted(set(prot_all.getChids()))
    if not protein_chains:
        raise ValueError("No protein chains found in structure")

    # If there is only one protein chain, use it
    if len(protein_chains) == 1:
        return protein_chains[0]

    # Multiple chains: find the one closest to the ligand
    lig_coords = ligand_sel.getCoords()
    if lig_coords is None or len(lig_coords) == 0:
        # Should not happen if ligand_sel is valid, but fallback to first
        return protein_chains[0]

    best_chain = protein_chains[0]
    best_min_d2 = float("inf")

    for ch in protein_chains:
        psel = structure.select(f"protein and chain {ch} and altloc _")
        if psel is None:
            psel = structure.select(f"protein and chain {ch}")
        if psel is None:
            continue
        pcoords = psel.getCoords()
        if pcoords is None or len(pcoords) == 0:
            continue

        # Minimum squared distance between any protein atom and any ligand atom
        d2 = ((pcoords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(axis=2)
        min_d2 = float(d2.min())
        if min_d2 < best_min_d2:
            best_min_d2 = min_d2
            best_chain = ch

    return best_chain

def extract_protein_and_ligand(
    pdb_id: str, 
    output_dir: Union[str, Path] = ".",
    ligand_resname: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    mmCIF-first receptor/ligand extraction:
    - Download mmCIF (ProDy fetchPDB if possible, else direct RCSB)
    - Parse with ProDy
    - Detect ligand instance + correct chain
    - Write `*_protein.pdb` and `*_ligand.pdb` via ProDy selections (required for AutoGrid/AutoDock)
    """
    outdir = Path(output_dir)
    structure_path = _download_mmcif(pdb_id, outdir)
    structure = _parse_mmcif(structure_path)

    protein_pdb = outdir / f"{pdb_id}_protein.pdb"
    ligand_pdb = outdir / f"{pdb_id}_ligand.pdb"

    if ligand_resname:
        lig_chain, lig_resname, lig_resnum, altloc_token = _pick_ligand_instance(structure, ligand_resname)
    else:
        lig_chain, lig_resname, lig_resnum, altloc_token = _pick_default_ligand(structure)

    ligand_sel = _select_ligand(structure, lig_chain, lig_resname, lig_resnum, altloc_token)

    detected_chain = _pick_receptor_chain_from_ligand(structure, ligand_sel)
    
    if lig_chain != detected_chain:
        # Common for mmCIF: ligand chain differs from polymer chain
        print(f"   Note: ligand '{lig_resname}' is in chain '{lig_chain}' (mmCIF asym-id); receptor protein chain is '{detected_chain}'.")
    else:
        print(f"   Note: using receptor protein chain '{detected_chain}'.")

    protein_sel = _select_protein(structure, detected_chain)
    writePDB(str(protein_pdb), protein_sel)
    writePDB(str(ligand_pdb), ligand_sel)

    return protein_pdb, ligand_pdb

class ReceptorPreparer:
    def __init__(self, autogrid_executable: str = "autogrid4", mk_prepare_receptor_executable: str = "mk_prepare_receptor.py", pdb2pqr_executable: str = "pdb2pqr"):
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

        if shutil.which("autogrid4") is None and self.autogrid_executable == "autogrid4":
            print("   Warning: autogrid4 not found in PATH. You may need to run 'module load autogrid' or provide the path.")

        self.mk_prepare_receptor_executable = shutil.which(mk_prepare_receptor_executable) or mk_prepare_receptor_executable
        self.pdb2pqr_executable = shutil.which(pdb2pqr_executable) or pdb2pqr_executable

    def prepare_receptor_and_grid(
        self, 
        pdb_id: str, 
        output_dir: Union[str, Path] = ".", 
        allow_bad_res: bool = False,
        ligand_resname: Optional[str] = None,
        protein_pdb_path: Optional[Union[str, Path]] = None,
        ligand_pdb_path: Optional[Union[str, Path]] = None,
        use_pdb2pqr: bool = True,
        ph: float = 7.4
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
            protein_pdb, ligand_pdb = extract_protein_and_ligand(pdb_id, output_dir=grid_dir, ligand_resname=ligand_resname)
        
        # 1.5. Optional: Run PDB2PQR to add hydrogens and optimize H-bond network
        pqr_path = None
        if use_pdb2pqr:
            # PDB2PQR is picky about ProDy remarks. Strip non-standard lines first.
            clean_protein_pdb = protein_pdb.parent / f"{protein_pdb.stem}_clean.pdb"
            try:
                with open(protein_pdb, "r") as f_in, open(clean_protein_pdb, "w") as f_out:
                    for line in f_in:
                        if line.startswith(("ATOM", "HETATM", "TER", "END")):
                            f_out.write(line)
                orig_protein_pdb = clean_protein_pdb # Keep reference for last ditch fallback
                protein_pdb = clean_protein_pdb
            except: 
                orig_protein_pdb = protein_pdb
        else:
            orig_protein_pdb = protein_pdb

        if use_pdb2pqr:
            fixed_pdb = protein_pdb.parent / f"{protein_pdb.stem}_fixed.pdb"
            pqr_path = fixed_pdb.with_suffix(".pqr")
            self._run_pdb2pqr(protein_pdb, fixed_pdb, ph=ph)
            if fixed_pdb.exists():
                protein_pdb = fixed_pdb
        
        # 2. Run mk_prepare_receptor.py
        base_name = f"rec_{pdb_id.lower()}"
        
        # Use PQR as input if available, as it avoids formatting issues in PDB fixed by PDB2PQR
        input_file = protein_pdb.name
        read_flag = "--read_pdb"
        extra_flags = []
        if pqr_path and pqr_path.exists():
            input_file = pqr_path.name
            read_flag = "--read_pqr"
            extra_flags = ["--charge_model", "read"]

        cmd = [
            self.mk_prepare_receptor_executable,
            read_flag, input_file,
            "-o", base_name,
            "-p", "-g",
            "--box_enveloping", str(ligand_pdb.name),
            "--padding", "5"
        ] + extra_flags
        
        if allow_bad_res:
            cmd.append("--allow_bad_res")
            
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(grid_dir), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   Warning: mk_prepare_receptor failed with {input_file} (code {result.returncode})")
            print(f"   Error: {result.stderr.strip().split('\n')[-1]}")
            
            # Fallback to PDB if PQR failed
            if read_flag == "--read_pqr" and protein_pdb.exists():
                print(f"   Attempting fallback to PDB: {protein_pdb.name}")
                cmd = [
                    self.mk_prepare_receptor_executable,
                    "--read_pdb", protein_pdb.name,
                    "-o", base_name,
                    "-p", "-g",
                    "--box_enveloping", str(ligand_pdb.name),
                    "--padding", "5"
                ]
                if allow_bad_res:
                    cmd.append("--allow_bad_res")
                
                print(f"   Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=str(grid_dir), capture_output=True, text=True)
                
            # If still failing, try original protein PDB (no PDB2PQR optimization)
            if result.returncode != 0 and orig_protein_pdb and orig_protein_pdb.exists() and orig_protein_pdb != protein_pdb:
                print(f"   Attempting fallback to ORIGINAL PDB: {orig_protein_pdb.name}")
                cmd = [
                    self.mk_prepare_receptor_executable,
                    "--read_pdb", orig_protein_pdb.name,
                    "-o", base_name,
                    "-p", "-g",
                    "--box_enveloping", str(ligand_pdb.name),
                    "--padding", "5"
                ]
                if allow_bad_res:
                    cmd.append("--allow_bad_res")
                
                print(f"   Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=str(grid_dir), capture_output=True, text=True)

            if result.returncode != 0:
                print(f"   Final Error: {result.stderr}")
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
        # Capture output to show on failure
        ag_result = subprocess.run(ag_cmd, cwd=str(grid_dir), capture_output=True, text=True)
        
        if ag_result.returncode != 0:
            print(f"   AutoGrid4 Output:\n{ag_result.stdout}")
            print(f"   AutoGrid4 Error:\n{ag_result.stderr}")
            raise GridPrepError(f"AutoGrid4 failed (returned {ag_result.returncode}). Check {glg_path} for details.")
            
        fld_path = grid_dir / f"{base_name}.maps.fld"
        return fld_path

    def _run_pdb2pqr(self, input_pdb: Path, output_pdb: Path, ph: float = 7.4):
        """
        Run PDB2PQR to add hydrogens and optimize the H-bond network (pH 7.4).
        """
        pqr_path = output_pdb.with_suffix(".pqr")
        
        # Use only filenames in the command because we set cwd to input_pdb.parent
        cmd = [
            self.pdb2pqr_executable,
            "--ff", "AMBER",
            "--titration-state-method", "propka",
            "--with-ph", str(ph),
            "--pdb-output", output_pdb.name,
            input_pdb.name,
            pqr_path.name
        ]
        
        print(f"Running PDB2PQR: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(input_pdb.parent))
            if result.returncode != 0:
                print(f"   PDB2PQR primary attempt failed (code {result.returncode}): {result.stderr}")
                
                # Check if it's a 'not found' error and try the specific environment suggested by the user
                if "not found" in result.stderr or result.returncode == 127:
                    activate_cmd = f"conda activate py312 && source .venv/bin/activate && pdb2pqr --ff AMBER --titration-state-method propka --with-ph {ph} --pdb-output {output_pdb.name} {input_pdb.name} {pqr_path.name}"
                    print(f"   Attempting with environment activation: {activate_cmd}")
                    # Note: conda activate requires shell=True and sourcing conda.sh which is complex.
                    # As a simpler alternative, try to locate the pdb2pqr in the project's .venv
                    venv_pdb2pqr = Path.cwd() / ".venv" / "bin" / "pdb2pqr"
                    if venv_pdb2pqr.exists():
                        cmd[0] = str(venv_pdb2pqr)
                        print(f"   Found venv pdb2pqr at: {venv_pdb2pqr}")
                        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(input_pdb.parent))
                    
            if result.returncode == 0:
                print(f"   PDB2PQR successful: {output_pdb.name}")
            else:
                print(f"   Warning: PDB2PQR failed. Receptor may lack hydrogens or proper protonation.")
        except Exception as e:
            print(f"   Error running PDB2PQR: {e}")

    # Keep compatibility with previous API if needed, but redirects to the new one
    def prepare_receptor(self, pdb_id: str, output_dir: Union[str, Path] = ".", allow_bad_res: bool = False) -> Path:
        # For compatibility, returns the pdbqt path
        grid_dir = Path(output_dir) / "grid"
        if not (grid_dir / f"{pdb_id}_receptor.pdbqt").exists():
            self.prepare_receptor_and_grid(pdb_id, output_dir, allow_bad_res)
        return grid_dir / f"{pdb_id}_receptor.pdbqt"

    def generate_grid(self, receptor_pdbqt: Union[str, Path], *args, **kwargs) -> Path:
        # For compatibility, returns the fld path
        # If it was already generated by prepare_receptor_and_grid, just return it
        fld_path = Path(receptor_pdbqt).with_suffix(".maps.fld")
        if fld_path.exists():
            return fld_path
        # Otherwise this indicates the old API flow was used, which we want to discourage
        raise NotImplementedError("Old generate_grid API is deprecated. Use prepare_receptor_and_grid.")
