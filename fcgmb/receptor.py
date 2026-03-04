import requests
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Union, Optional, Tuple, List
import numpy as np

# ProDy
from prody import fetchPDB, parseMMCIF, writePDB, confProDy

confProDy(verbosity="error")


class PDBDownloadError(Exception):
    """Raised when a PDB file cannot be downloaded from RCSB."""

    pass


class LigandNotFoundError(Exception):
    """Raised when the specified ligand resname is not found in the structure."""

    pass


class GridPrepError(Exception):
    """Raised when mk_prepare_receptor or AutoGrid fails."""

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
        structure_path = fetchPDB(
            pdb_id, folder=str(output_dir), compressed=False, format="cif"
        )
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


def _pick_ligand_instance(
    structure, ligand_resname: str
) -> Tuple[str, str, int, Optional[str]]:
    """
    Find the first instance of a ligand resname and return (chain_id, resname, resnum, altloc_token).
    """
    all_ligands = structure.select(f"resname {ligand_resname}")
    if all_ligands is None:
        raise LigandNotFoundError(
            f"Ligand with resname {ligand_resname} not found in structure"
        )

    altloc_token = _choose_altloc_token(all_ligands)
    ligands_for_res = (
        all_ligands.select(f"altloc {altloc_token}") if altloc_token else all_ligands
    )
    ligand_res = ligands_for_res.getHierView().iterResidues().__next__()
    return (
        ligand_res.getChid(),
        ligand_res.getResname(),
        ligand_res.getResnum(),
        altloc_token,
    )


def _select_protein(structure, chain_ids: List[str]):
    """
    Select protein atoms for one or more chains. Prefer blank altloc (to avoid duplicates),
    but fall back to any altloc if blank selection is empty.
    """
    chain_str = " ".join(chain_ids)
    sel = structure.select(f"protein and chain {chain_str} and altloc _")
    if sel is None:
        sel = structure.select(f"protein and chain {chain_str}")
    if sel is None:
        raise ValueError(f"No protein atoms found in chains {chain_str}")
    return sel


def _select_ligand(
    structure, chain_id: str, resname: str, resnum: int, altloc_token: Optional[str]
):
    """
    Select ligand atoms for a specific residue instance.
    """
    if altloc_token:
        sel = structure.select(
            f"chain {chain_id} and resname {resname} and resnum {resnum} and altloc {altloc_token}"
        )
        if sel is not None:
            return sel
    sel = structure.select(
        f"chain {chain_id} and resname {resname} and resnum {resnum}"
    )
    if sel is None:
        raise ValueError(f"Failed to select ligand {resname} {chain_id} {resnum}")
    return sel


def _pick_receptor_chains_from_ligand(
    structure, ligand_sel, distance_threshold: float = 5.0
) -> List[str]:
    """
    Identify protein chains within a distance threshold of the ligand.
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
        return [protein_chains[0]]

    # Multiple chains: find all within distance threshold
    lig_coords = ligand_sel.getCoords()
    if lig_coords is None or len(lig_coords) == 0:
        return [protein_chains[0]]

    nearby_chains = []

    # Also find the single closest chain as a fallback if none are within threshold
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

        if min_d2 < (distance_threshold**2):
            nearby_chains.append(ch)

        if min_d2 < best_min_d2:
            best_min_d2 = min_d2
            best_chain = ch

    if not nearby_chains:
        return [best_chain]

    return nearby_chains


def extract_protein_and_ligand(
    pdb_id: str, ligand_resname: str, output_dir: Union[str, Path] = "."
) -> Tuple[Path, Path]:
    """
    mmCIF-first receptor/ligand extraction:
    - Download mmCIF (ProDy fetchPDB if possible, else direct RCSB)
    - Parse with ProDy
    - Detect ligand instance + correct chains
    - Write `*_protein.pdb` and `*_ligand.pdb` via ProDy selections
    """
    if not ligand_resname:
        raise ValueError("ligand_resname must be specified.")

    outdir = Path(output_dir)
    structure_path = _download_mmcif(pdb_id, outdir)
    structure = _parse_mmcif(structure_path)

    protein_pdb = outdir / f"{pdb_id}_protein.pdb"
    ligand_pdb = outdir / f"{pdb_id}_ligand.pdb"

    lig_chain, lig_resname, lig_resnum, altloc_token = _pick_ligand_instance(
        structure, ligand_resname
    )
    ligand_sel = _select_ligand(
        structure, lig_chain, lig_resname, lig_resnum, altloc_token
    )

    detected_chains = _pick_receptor_chains_from_ligand(structure, ligand_sel)

    print(
        f"   Note: ligand '{lig_resname}' is in chain '{lig_chain}'; receptor protein chains: {', '.join(detected_chains)}."
    )

    protein_sel = _select_protein(structure, detected_chains)
    writePDB(str(protein_pdb), protein_sel)
    writePDB(str(ligand_pdb), ligand_sel)

    return protein_pdb, ligand_pdb


class ReceptorPreparer:
    def __init__(
        self,
        autogrid_executable: str = "autogrid4",
        mk_prepare_receptor_executable: str = "mk_prepare_receptor.py",
        reduce2_executable: str = "mmtbx.reduce2",
    ):
        self.autogrid_executable = shutil.which(autogrid_executable)
        if self.autogrid_executable is None:
            raise FileNotFoundError(
                f"Executable '{autogrid_executable}' not found in PATH"
            )

        self.mk_prepare_receptor_executable = shutil.which(
            mk_prepare_receptor_executable
        )
        if self.mk_prepare_receptor_executable is None:
            raise FileNotFoundError(
                f"Executable '{mk_prepare_receptor_executable}' not found in PATH"
            )

        # Preferred hydrogenation tool.
        self.reduce2_executable = shutil.which(reduce2_executable)
        if self.reduce2_executable is None:
            raise FileNotFoundError(
                f"Executable '{reduce2_executable}' not found in PATH"
            )

    def get_receptor_and_ligand_pdb(
        self,
        pdb_id: str,
        output_dir: Path,
        ligand_resname: str,
        protein_pdb_path: Optional[Union[str, Path]] = None,
        ligand_pdb_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """Obtain protein and ligand PDB files."""
        if protein_pdb_path and ligand_pdb_path:
            protein_pdb_path = Path(protein_pdb_path)
            ligand_pdb_path = Path(ligand_pdb_path)

            if not protein_pdb_path.exists():
                raise FileNotFoundError(f"Protein PDB not found: {protein_pdb_path}")
            if not ligand_pdb_path.exists():
                raise FileNotFoundError(f"Ligand PDB not found: {ligand_pdb_path}")

            protein_pdb = output_dir / protein_pdb_path.name
            ligand_pdb = output_dir / ligand_pdb_path.name
            shutil.copy2(protein_pdb_path, protein_pdb)
            shutil.copy2(ligand_pdb_path, ligand_pdb)
            print(
                f"Using provided protein ({protein_pdb.name}) and ligand ({ligand_pdb.name})"
            )
            return protein_pdb, ligand_pdb
        else:
            return extract_protein_and_ligand(
                pdb_id, ligand_resname, output_dir=output_dir
            )

    def run_reduce2(self, protein_pdb: Path) -> Path:
        """Run mmtbx.reduce2 to add hydrogens (preferred)."""
        if self.reduce2_executable is None:
            raise FileNotFoundError("Executable 'mmtbx.reduce2' not found in PATH")

        # Strip non-standard lines to avoid downstream parsing issues.
        clean_protein_pdb = protein_pdb.parent / f"{protein_pdb.stem}_clean.pdb"
        with open(protein_pdb, "r") as f_in, open(clean_protein_pdb, "w") as f_out:
            for line in f_in:
                if line.startswith(("ATOM", "HETATM", "TER", "END")):
                    f_out.write(line)

        reduced_pdb = clean_protein_pdb.parent / f"{clean_protein_pdb.stem}H.pdb"
        cmd = [
            self.reduce2_executable,
            clean_protein_pdb.name,
            "--overwrite",
            "--quiet",
        ]
        print(f"Running reduce2: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(protein_pdb.parent)
        )
        if result.returncode != 0:
            print(
                f"   Warning: reduce2 failed (code {result.returncode}): {result.stderr}"
            )
            raise GridPrepError("reduce2 failed while adding hydrogens.")

        if reduced_pdb.exists():
            return reduced_pdb

        print(
            "   Warning: reduce2 produced no output; using cleaned PDB without hydrogens."
        )
        return clean_protein_pdb

    def run_mk_prepare_receptor(
        self,
        receptor_input: Path,
        ligand_pdb: Path,
        base_name: str,
        output_dir: Path,
        allow_bad_res: bool = False,
    ) -> Path:
        """Run mk_prepare_receptor.py to create PDBQT and GPF."""

        cmd = [
            self.mk_prepare_receptor_executable,
            "--read_pdb",
            receptor_input.name,
            "-o",
            base_name,
            "-p",
            "-g",
            "--box_enveloping",
            ligand_pdb.name,
            "--padding",
            "5",
        ]

        if allow_bad_res:
            cmd.append("--allow_bad_res")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(output_dir), capture_output=True, text=True
        )

        if result.returncode != 0:
            err_lines = [
                line.strip()
                for line in (result.stderr or "").splitlines()
                if line.strip()
            ]
            summary = err_lines[-1] if err_lines else "Unknown error"
            print(f"   Error: {summary}")
            raise GridPrepError(
                f"Receptor preparation failed (mk_prepare_receptor.py returned {result.returncode})."
            )

        return output_dir / f"{base_name}.gpf"

    def run_autogrid(self, gpf_path: Path, output_dir: Path) -> Path:
        """Run AutoGrid4."""
        base_name = gpf_path.stem
        glg_path = output_dir / f"{base_name}.glg"

        print(f"Running AutoGrid4 for {gpf_path.name}...")

        ag_cmd = [self.autogrid_executable, "-p", gpf_path.name, "-l", glg_path.name]
        ag_result = subprocess.run(
            ag_cmd, cwd=str(output_dir), capture_output=True, text=True
        )

        if ag_result.returncode != 0:
            print(f"   AutoGrid4 Error:\n{ag_result.stderr}")
            raise GridPrepError(f"AutoGrid4 failed (returned {ag_result.returncode}).")

        return output_dir / f"{base_name}.maps.fld"

    def prepare_receptor_and_grid(
        self,
        pdb_id: str,
        ligand_resname: str,
        output_dir: Union[str, Path] = ".",
        allow_bad_res: bool = False,
        protein_pdb_path: Optional[Union[str, Path]] = None,
        ligand_pdb_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Convenience method to run the full preparation pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain PDBs
        protein_pdb, ligand_pdb = self.get_receptor_and_ligand_pdb(
            pdb_id, output_dir, ligand_resname, protein_pdb_path, ligand_pdb_path
        )

        # 2. Add hydrogens
        receptor_input = self.run_reduce2(protein_pdb)

        # 3. Run mk_prepare_receptor
        base_name = f"rec_{pdb_id.lower()}"
        try:
            gpf_path = self.run_mk_prepare_receptor(
                receptor_input, ligand_pdb, base_name, output_dir, allow_bad_res
            )
        except GridPrepError as e:
            raise e

        # 4. Run AutoGrid
        return self.run_autogrid(gpf_path, output_dir)
