import os
import shutil
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union, Tuple
import gzip
import time

import polars as pl
from molscrub import Scrub
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem

class AutoDockGPUOracle:
    def __init__(
        self,
        receptor_file: Union[str, Path],
        adgpu_executable: str = "adgpu",
        n_poses: int = 10,
        n_cpus: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        ph_low: float = 6.4,
        ph_high: float = 8.4
    ):
        """
        Oracle for scoring molecules using AutoDock-GPU.

        Args:
            receptor_file: Path to the receptor maps.fld file.
            adgpu_executable: Path to the adgpu executable or name if in PATH.
            n_poses: Number of poses to generate per ligand.
            n_cpus: Number of CPUs for parallel ligand preparation. Defaults to all available.
            save_dir: Directory to save persistent PDBQT and DLG files.
            ph_low: Minimum pH for protonation state enumeration.
            ph_high: Maximum pH for protonation state enumeration.
        """
        self.receptor_file = Path(receptor_file).resolve()
        self.adgpu_executable = adgpu_executable
        self.n_poses = n_poses
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Initialize Scrubber
        self.scrub = Scrub(ph_low=ph_low, ph_high=ph_high)
        
        # Tracking results
        self.results_df = None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "ligands").mkdir(exist_ok=True)
            (self.save_dir / "results").mkdir(exist_ok=True)

        # Validation
        if not self.receptor_file.exists():
            raise FileNotFoundError(f"Receptor file not found: {self.receptor_file}")
            
        # Check if adgpu is callable
        resolved_exe = shutil.which(self.adgpu_executable)
        if resolved_exe is None:
             # If not in path, check if it is a direct path
            if Path(self.adgpu_executable).exists() and os.access(self.adgpu_executable, os.X_OK):
                 resolved_exe = self.adgpu_executable
            else:
                 raise FileNotFoundError(f"AutoDock-GPU executable not found or not executable: {self.adgpu_executable}")
        
        self.adgpu_executable = resolved_exe

    def __call__(self, smiles: str) -> float:
        """
        Score a single SMILES string. Returns the best docking score (lowest energy).
        Returns high value (e.g. 999.9) if docking fails.
        """
        results = self.score_batch([smiles])
        return results[0] if results else 999.9

    def score_batch(self, smiles_list: List[str], batch_size: int = 100) -> List[float]:
        """
        Score a batch of SMILES strings.
        """
        all_results = []
        
        # Process in chunks to avoid overwhelming file system or args
        for i in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(smiles_list) - 1) // batch_size + 1} ({len(chunk)} compounds)...")
            chunk_results = self._process_chunk(chunk, chunk_idx=i // batch_size)
            all_results.extend(chunk_results)
            
        # Store detailed results in a DataFrame
        self.results_df = pl.DataFrame(all_results)
        
        # Return just the scores for compatibility
        return self.results_df.get_column("docking_score").to_list()

    def _prepare_single_ligand(self, args: Tuple[str, int, Path]) -> List[Tuple[str, str]]:
        """
        Prepare multiple ligand PDBQTs (states/conformers) for a single SMILES using Scrub and Meeko.
        Returns a list of (pdbqt_string, suffix).
        """
        smiles, idx, pdbqt_dir = args
        results = []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            # 1. Use Scrub to generate tautomers/protonation states/stereoisomers
            try:
                mol_states = list(self.scrub(mol))
            except Exception as e:
                print(f"Scrub failed for {smiles}: {e}")
                mol_states = [mol] # Fallback to original mol
            
            preparator = MoleculePreparation()
            
            state_counter = 0
            for mol_state in mol_states:
                try:
                    # Basic cleanup and 3D embedding for each state
                    mol_state = Chem.AddHs(mol_state)
                    if AllChem.EmbedMolecule(mol_state, randomSeed=42) != 0:
                        AllChem.Compute2DCoords(mol_state)
                    
                    # 2. Use Meeko to prepare each state
                    mol_setups = preparator.prepare(mol_state)
                    
                    for mol_setup in mol_setups:
                        # Use PDBQTWriterLegacy for compatibility with ADGPU
                        pdbqt_string, ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)
                        
                        if ok:
                            results.append((pdbqt_string, f"s{state_counter}"))
                            state_counter += 1
                        else:
                            print(f"Meeko write failed for state {state_counter} of {smiles}: {error_msg}")
                except Exception as e:
                    print(f"Error preparing state {state_counter} for {smiles}: {e}")
                    continue
                    
            return results
        except Exception as e:
            print(f"Error in _prepare_single_ligand for {smiles}: {e}")
            return []

    def _process_chunk(self, smiles_list: List[str], chunk_idx: int) -> List[dict]:
        # Create temp dir for this chunk
        with tempfile.TemporaryDirectory(prefix=f"adgpu_chunk_{chunk_idx}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdbqt_dir = tmp_path / "ligands"
            pdbqt_dir.mkdir()
            
            # Prepare ligands in parallel
            prep_args = [(smi, idx, pdbqt_dir) for idx, smi in enumerate(smiles_list)]
            
            with multiprocessing.Pool(processes=self.n_cpus) as pool:
                # Each element in pdbqt_data is a list of (pdbqt_string, suffix)
                pdbqt_data = pool.map(self._prepare_single_ligand, prep_args)
            
            # Map from SMILES index to list of actual written PDBQT file names (relative to tmp_path)
            idx_to_pdbqts = {}
            for i, states in enumerate(pdbqt_data):
                if not states:
                    continue
                
                written_files = []
                for pdbqt_string, suffix in states:
                    fname = f"lig_{i}_{suffix}.pdbqt"
                    fpath = pdbqt_dir / fname
                    fpath.write_text(pdbqt_string)
                    written_files.append(f"ligands/{fname}")
                    
                    # Persistent storage if requested
                    if self.save_dir:
                        shutil.copy2(fpath, self.save_dir / "ligands" / f"chunk_{chunk_idx}_{fname}")
                
                idx_to_pdbqts[i] = written_files

            if not idx_to_pdbqts:
                return [{"smiles": s, "docking_score": 999.9, "dlg_path": None} for s in smiles_list]

            # Create filelist
            filelist_path = tmp_path / "filelist.txt"
            
            # Symlink receptor maps and parameter files to the temp directory
            receptor_dir = self.receptor_file.parent
            for ref_file in receptor_dir.iterdir():
                if ref_file.is_file():
                    target = tmp_path / ref_file.name
                    if not target.exists():
                        try:
                            os.symlink(ref_file.resolve(), target)
                        except OSError:
                            shutil.copy2(ref_file, target)
            
            with open(filelist_path, "w") as f:
                f.write(f"{self.receptor_file.name}\n")
                for i, written_files in idx_to_pdbqts.items():
                    for rel_path in written_files:
                        f.write(f"{rel_path}\n")
                        # Output name is just the stem of the ligand file
                        f.write(f"{Path(rel_path).stem}\n")

            # Run ADGPU
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': str(multiprocessing.cpu_count()),
                'OMP_PROC_BIND': 'true',
                'OMP_PLACES': 'cores'
            })

            cmd = [
                self.adgpu_executable,
                "--filelist", "filelist.txt",
                "--nrun", str(self.n_poses),
                "--xmloutput", "0",
                "--dlgoutput", "1"
            ]
            
            try:
                result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, env=env, text=True)
                if result.returncode != 0:
                    print(f"ADGPU failed for chunk {chunk_idx}:")
                    print(f"Stdout: {result.stdout}")
                    print(f"Stderr: {result.stderr}")
                    if "cuda" in result.stderr.lower() or "gpu" in result.stderr.lower():
                        raise RuntimeError(f"AutoDock-GPU CUDA/GPU error: {result.stderr}")
                elif "Error" in result.stdout:
                    print(f"ADGPU reported errors in stdout for chunk {chunk_idx}:")
                    print(result.stdout)
            except Exception as e:
                print(f"Exception running ADGPU: {e}")

            # Parse results
            chunk_results = []
            
            for i, smi in enumerate(smiles_list):
                best_score_for_smiles = 999.9
                best_dlg_for_smiles = None
                
                if i in idx_to_pdbqts:
                    for rel_path in idx_to_pdbqts[i]:
                        stem = Path(rel_path).stem
                        dlg_path = tmp_path / f"{stem}.dlg"
                        
                        if dlg_path.exists():
                            persistent_dlg = None
                            # Save DLG if requested
                            if self.save_dir:
                                persistent_dlg = self.save_dir / "results" / f"chunk_{chunk_idx}_{stem}.dlg"
                                shutil.copy2(dlg_path, persistent_dlg)
                            
                            try:
                                val = self._parse_dlg(dlg_path)
                                if val is not None and val < best_score_for_smiles:
                                    best_score_for_smiles = val
                                    best_dlg_for_smiles = str(persistent_dlg) if persistent_dlg else str(dlg_path)
                            except Exception as e:
                                print(f"Error parsing {dlg_path}: {e}")
                
                chunk_results.append({
                    "smiles": smi,
                    "docking_score": best_score_for_smiles,
                    "dlg_path": best_dlg_for_smiles
                })
            
            return chunk_results

    def _parse_dlg(self, dlg_path: Path) -> Optional[float]:
        """
        Robust parsing of best energy using Meeko's PDBQTMolecule.
        """
        try:
            # PDBQTMolecule.from_file can parse .dlg files
            pdbqt_mol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=True, skip_typing=True)
            
            # _pose_data contains energies
            # For ADGPU, the poses are already sorted by rank in clusters or we can just find the minimum free_energy
            if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
                energies = pdbqt_mol._pose_data["free_energies"]
                if energies:
                    return float(min(energies))
            
            # Fallback to manual parsing if Meeko structure is different or fails
            best_score = 999.9
            found = False
            with open(dlg_path, "r") as f:
                for line in f:
                    if "|    1 |" in line or "   1 |" in line:
                        parts = line.split("|")
                        if len(parts) >= 3:
                            try:
                                score = float(parts[2].strip())
                                if score < best_score:
                                    best_score = score
                                    found = True
                            except ValueError:
                                continue
            return best_score if found else None
            
        except Exception as e:
            # Last resort fallback if Meeko fails completely
            try:
                with open(dlg_path, "r") as f:
                    for line in f:
                        if "Rank | Binding Energy" in line:
                            # Skip header and separator
                            next(f)
                            next(f)
                            line = next(f)
                            parts = line.split("|")
                            if len(parts) >= 3:
                                return float(parts[2].strip())
            except:
                pass
            print(f"Warning: Failed to parse {dlg_path} using all methods: {e}")
            return None

    def save_best_poses_sdf(self, output_path: Union[str, Path]):
        """
        Extract the best pose from each successful docking run and save to an SDF.
        """
        if self.results_df is None:
            print("No results to save. Run score_batch first.")
            return

        print(f"Generating best poses SDF at {output_path}...")
        writer = Chem.SDWriter(str(output_path))
        count = 0
        
        for row in self.results_df.iter_rows(named=True):
            if row['docking_score'] >= 999.0 or row['dlg_path'] is None:
                continue
            
            try:
                # Load DLG with Meeko
                pdbqt_mol = PDBQTMolecule.from_file(row['dlg_path'], is_dlg=True, skip_typing=True)
                # Create RDKit molecules for all poses, Meeko keeps them in order of rank
                rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
                
                if rdkit_mols:
                    best_mol = rdkit_mols[0]
                    # Add metadata
                    best_mol.SetProp("_Name", row['molecule_chembl_id'])
                    best_mol.SetProp("smiles", row['smiles'])
                    best_mol.SetProp("docking_score", str(row['docking_score']))
                    best_mol.SetProp("dlg_path", row['dlg_path'])
                    writer.write(best_mol)
                    count += 1
            except Exception as e:
                print(f"Failed to extract pose for {row['smiles']}: {e}")

        writer.close()
        print(f"Successfully saved {count} best poses to {output_path}")
