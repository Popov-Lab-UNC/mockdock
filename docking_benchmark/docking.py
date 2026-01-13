import os
import shutil
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union, Tuple
import gzip
import time
import math

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
        ph_high: float = 8.4,
        reference_ligand_path: Optional[Union[str, Path]] = None,
        fragment_smiles: Optional[str] = None,
        rmsd_threshold: float = 2.0,
        generate_isomers: bool = True
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
            reference_ligand_path: Path to the reference ligand PDB for RMSD calculation.
            fragment_smiles: SMILES string for the fragment constraint (no exit vectors).
            rmsd_threshold: RMSD threshold for the fragment constraint.
            generate_isomers: Whether to generate stereoisomers (and use Scrub for them).
        """
        self.receptor_file = Path(receptor_file).resolve()
        self.adgpu_executable = adgpu_executable
        self.n_poses = n_poses
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.ph_low = ph_low
        self.ph_high = ph_high
        self.generate_isomers = generate_isomers
        
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

        # Fragment constraint setup
        self.reference_ligand_path = reference_ligand_path
        self.fragment_smiles = fragment_smiles
        self.rmsd_threshold = rmsd_threshold
        self.ref_mol = None
        self.fragment_mol = None

        if self.reference_ligand_path and self.fragment_smiles:
            self.reference_ligand_path = Path(self.reference_ligand_path)
            if not self.reference_ligand_path.exists():
                raise FileNotFoundError(f"Reference ligand not found: {self.reference_ligand_path}")

            # Load reference ligand
            if self.reference_ligand_path.suffix.lower() == ".sdf":
                suppl = Chem.SDMolSupplier(str(self.reference_ligand_path), removeHs=False)
                self.ref_mol = next(iter(suppl), None)
            else:
                self.ref_mol = Chem.MolFromPDBFile(str(self.reference_ligand_path), removeHs=False)

            if self.ref_mol is None:
                raise ValueError(f"Could not load reference ligand from {self.reference_ligand_path}")

            # Try parsing as SMILES first, then SMARTS as fallback
            self.fragment_mol = Chem.MolFromSmiles(self.fragment_smiles)
            if self.fragment_mol is None:
                # Try SMARTS as fallback (useful for MCS patterns or query molecules)
                self.fragment_mol = Chem.MolFromSmarts(self.fragment_smiles)
                if self.fragment_mol is None:
                    raise ValueError(f"Invalid fragment SMILES/SMARTS string: {self.fragment_smiles}")

            # Verify reference matches Fragment
            if not self._get_robust_match(self.ref_mol, self.fragment_mol):
                print(f"WARNING: Reference ligand ({self.reference_ligand_path.name}) does not match fragment SMILES!")
                print("This is likely due to missing/incorrect bond orders in the PDB file.")
                print("RMSD filtering will fail for all compounds.")
            else:
                print("Reference ligand loaded and matches fragment SMILES.")


    def __call__(self, smiles: str) -> float:
        """
        Score a single SMILES string. Returns the best docking score (lowest energy).
        Returns high value (e.g. 999.9) if docking fails or fragment match checks fail.
        """
        results = self.score_batch([smiles])
        return results.get(smiles, 999.9)

    def score_batch(self, smiles_list: List[str], batch_size: int = 100) -> dict[str, float]:
        """
        Score a batch of SMILES strings. Returns a dictionary {smiles: score}.
        Only docks molecules that match the 2D fragment constraint if provided.
        """
        # 2D Pre-filtering: Only dock compounds that match the fragment
        valid_smiles = []
        skipped_results = []
        
        if self.fragment_mol:
            print(f"-> Performing 2D fragment pre-filtering for {len(smiles_list)} compounds...")
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.HasSubstructMatch(self.fragment_mol):
                    valid_smiles.append(smi)
                else:
                    # Record skipped compound
                    skipped_results.append({
                        "smiles": smi,
                        "docking_score": float('nan'),
                        "score_valid": float('nan'),
                        "score_best_any": float('nan'),
                        "dlg_path": None,
                        "dlg_path_valid": None,
                        "dlg_path_any": None,
                        "valid_pose_found": False,
                        "fragment_precheck": False,
                        "n_conformers": 0,
                        "skip_reason": "2D fragment mismatch" if mol else "Invalid SMILES"
                    })
            
            print(f"   Matches Fragment (2D): {len(valid_smiles)}/{len(smiles_list)} compounds")
            if skipped_results:
                print(f"   Skipping {len(skipped_results)} compounds that do not match the fragment.")
        else:
            valid_smiles = smiles_list

        all_results = skipped_results
        
        # Process valid smiles in chunks
        if valid_smiles:
            for i in range(0, len(valid_smiles), batch_size):
                chunk = valid_smiles[i : i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(valid_smiles) - 1) // batch_size + 1} ({len(chunk)} compounds)...", flush=True)
                chunk_results = self._process_chunk(chunk, chunk_idx=i // batch_size)
                all_results.extend(chunk_results)
            
        # Store detailed results in a DataFrame
        self.results_df = pl.DataFrame(all_results)
        
        # Return dictionary mapping smiles to best docking score
        df_scores = self.results_df.select(["smiles", "docking_score"])

        # Create a boolean mask for valid scores (not null and not NaN)
        valid_scores_mask = pl.col("docking_score").is_not_null() & pl.col("docking_score").is_not_nan()

        # Create the dictionary using a conditional expression
        smile_to_score_df = df_scores.with_columns(
            pl.when(valid_scores_mask)
            .then(pl.col("docking_score"))
            .otherwise(pl.lit(float('nan')))
            .alias("docking_score")
        )

        return dict(zip(smile_to_score_df["smiles"], smile_to_score_df["docking_score"]))

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
            
            # Check for explicit stereochemistry
            has_stereo = False
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            if centers:
                has_stereo = True

            # Determine if we should use Scrub for isomers or try to preserve input
            # Logic: If generate_isomers is False AND stereo is explicitly defined, we want to AVOID generating stereoisomers.
            # Scrub generates states. We can filter them.

            # Initialize Scrubber locally
            scrubber = Scrub(ph_low=self.ph_low, ph_high=self.ph_high)
            
            mol_states = []

            try:
                # If we want to restrict isomers (user flag or explicit stereo), we need to be careful with Scrub
                use_scrub_generated = True

                scrub_results = list(scrubber(mol))

                if (not self.generate_isomers) and has_stereo:
                    # Filter: Keep only those matching original stereo
                    filtered_states = []
                    # Get original tags (atom idx -> tag)
                    orig_tags = dict(centers)

                    for s_mol in scrub_results:
                        s_centers = Chem.FindMolChiralCenters(s_mol, includeUnassigned=False)
                        s_tags = dict(s_centers)

                        match = True
                        for idx, tag in orig_tags.items():
                            # Note: Scrub might reorder atoms?
                            # If atoms are reordered, index matching is invalid.
                            # Assuming Scrub preserves atom ordering or we rely on InChI/SMILES equality?
                            # SMILES with chirality should match.
                            if idx not in s_tags or s_tags[idx] != tag:
                                match = False
                                break
                        if match:
                            filtered_states.append(s_mol)

                    if filtered_states:
                        mol_states = filtered_states
                    else:
                        # Fallback: Just use original mol (with Hs)
                        mol_states = [mol]
                else:
                    mol_states = scrub_results

            except Exception as e:
                print(f"Scrub failed for {smiles}: {e}", flush=True)
                mol_states = [mol] # Fallback to original mol
            
            # Limit states
            if len(mol_states) > 16:
                print(f"Warning: {smiles} produced {len(mol_states)} states. Limiting to 16.", flush=True)
                mol_states = mol_states[:16]

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
                            print(f"Meeko write failed for state {state_counter} of {smiles}: {error_msg}", flush=True)
                        
                        if state_counter >= 32:
                            break
                    if state_counter >= 32:
                        break
                except Exception as e:
                    print(f"Error preparing state {state_counter} for {smiles}: {e}", flush=True)
                    continue
                    
            return results
        except Exception as e:
            print(f"Error in _prepare_single_ligand for {smiles}: {e}", flush=True)
            return []

    def _get_robust_match(self, target_mol: Chem.Mol, query_mol: Chem.Mol) -> Tuple[int]:
        """
        Attempt to find substructure match robust to tautomers/bond orders.
        """
        # 1. Try exact match
        match = target_mol.GetSubstructMatch(query_mol)
        if match:
            return match

        # 2. Try match with bond orders unspecified (tautomer-like robustness)
        # Create a query where bond orders are generic
        try:
            params = Chem.AdjustQueryParameters()
            params.adjustDegree = False
            params.adjustHeavyDegree = False
            params.makeBondsGeneric = True # Matches any bond order
            params.aromatizeIfPossible = True
            
            # Use AdjustQueryProperties to create a loose query from the fragment
            loose_query = Chem.AdjustQueryProperties(query_mol, params)
            
            match = target_mol.GetSubstructMatch(loose_query)
            if match:
                return match
        except Exception:
            pass

        return ()

    def _calculate_rmsd(self, probe_mol: Chem.Mol) -> float:
        """
        Calculate RMSD of the fragment between probe_mol and self.ref_mol.
        """
        if self.ref_mol is None or self.fragment_mol is None:
            return 0.0 # No constraint, RMSD is 0

        # Find matches using robust matcher
        ref_match = self._get_robust_match(self.ref_mol, self.fragment_mol)
        probe_match = self._get_robust_match(probe_mol, self.fragment_mol)

        if not ref_match or not probe_match:
            return 999.9 # Constraint not matched in topology

        # Get coordinates
        ref_conf = self.ref_mol.GetConformer()
        probe_conf = probe_mol.GetConformer()

        ref_coords = []
        probe_coords = []

        for idx in ref_match:
            pos = ref_conf.GetAtomPosition(idx)
            ref_coords.append((pos.x, pos.y, pos.z))

        for idx in probe_match:
            pos = probe_conf.GetAtomPosition(idx)
            probe_coords.append((pos.x, pos.y, pos.z))

        # Calculate RMSD
        # Manual RMSD to avoid alignment (we want absolute position check)
        sq_diff = 0
        for (rx, ry, rz), (px, py, pz) in zip(ref_coords, probe_coords):
            sq_diff += (rx - px)**2 + (ry - py)**2 + (rz - pz)**2

        rmsd = math.sqrt(sq_diff / len(ref_coords))
        return rmsd

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
                return [{
                    "smiles": s,
                    "docking_score": float('nan'),
                    "score_valid": float('nan'),
                    "score_best_any": float('nan'),
                    "dlg_path": None,
                    "valid_pose_found": False,
                    "fragment_precheck": False,
                    "n_conformers": 0
                } for s in smiles_list]

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
            # Limit OpenMP threads to avoid overhead on massive nodes
            n_threads = min(8, multiprocessing.cpu_count())
            env.update({
                'OMP_NUM_THREADS': str(n_threads),
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
                best_valid_score = float('nan')
                best_valid_dlg_path = None
                best_any_score = float('nan')
                best_any_dlg_path = None

                valid_pose_found = False
                
                # Check 2D match
                fragment_precheck = False
                mol = Chem.MolFromSmiles(smi)
                if mol and self.fragment_mol and mol.HasSubstructMatch(self.fragment_mol):
                    fragment_precheck = True
                elif not self.fragment_mol:
                    fragment_precheck = True # No filter

                if i in idx_to_pdbqts:
                    for rel_path in idx_to_pdbqts[i]:
                        stem = Path(rel_path).stem
                        dlg_path = tmp_path / f"{stem}.dlg"
                        
                        if dlg_path.exists():
                            persistent_dlg = None
                            if self.save_dir:
                                persistent_dlg = self.save_dir / "results" / f"chunk_{chunk_idx}_{stem}.dlg"
                                shutil.copy2(dlg_path, persistent_dlg)
                            
                            # Use extended parsing to handle constraints
                            try:
                                best_score, passed_constraint, best_mol, best_invalid_score, best_invalid_mol = self._parse_and_filter(dlg_path, persistent_dlg, smi)

                                cur_dlg_path_str = str(persistent_dlg) if persistent_dlg else str(dlg_path)

                                if passed_constraint:
                                    if math.isnan(best_valid_score) or best_score < best_valid_score:
                                        best_valid_score = best_score
                                        best_valid_dlg_path = cur_dlg_path_str
                                        valid_pose_found = True

                                # Track best ANY score
                                if not math.isnan(best_score): # If valid found, it's also a score
                                    if math.isnan(best_any_score) or best_score < best_any_score:
                                        best_any_score = best_score
                                        best_any_dlg_path = cur_dlg_path_str

                                if not math.isnan(best_invalid_score):
                                    if math.isnan(best_any_score) or best_invalid_score < best_any_score:
                                        best_any_score = best_invalid_score
                                        best_any_dlg_path = cur_dlg_path_str

                            except Exception as e:
                                print(f"Error parsing {dlg_path}: {e}")
                
                # Selection Logic:
                # If valid pose found (RMSD < threshold), use its score.
                # Else, use the overall best score (best_any_score).
                final_score = best_valid_score if valid_pose_found else best_any_score
                final_dlg_path = best_valid_dlg_path if valid_pose_found else best_any_dlg_path

                # Count PDBQTs (states) generated
                n_states = len(idx_to_pdbqts.get(i, []))

                chunk_results.append({
                    "smiles": smi,
                    "docking_score": final_score,         # The primary score (selected)
                    "score_valid": best_valid_score,      # Specific valid score
                    "score_best_any": best_any_score,     # Overall best
                    "dlg_path": final_dlg_path,           # Path to the DLG of selected score
                    "dlg_path_valid": best_valid_dlg_path,
                    "dlg_path_any": best_any_dlg_path,
                    "valid_pose_found": valid_pose_found, # Boolean flag
                    "fragment_precheck": fragment_precheck,
                    "n_conformers": n_states,                 # Metadata: Number of states/conformers docked
                    "skip_reason": None
                })
            
            return chunk_results

    def _parse_and_filter(self, dlg_path: Path, persistent_dlg: Optional[Path], smiles: str) -> Tuple[float, bool, Optional[Chem.Mol], float, Optional[Chem.Mol]]:
        """
        Parse DLG, filter poses by RMSD if applicable, and return (best_valid_score, passed_constraint, best_mol, best_invalid_score, best_invalid_mol).
        """
        # If no filtering needed, use fast path
        if not self.reference_ligand_path or not self.fragment_smiles:
             val = self._parse_dlg_simple(dlg_path)
             if val is not None:
                 return val, True, None, float('nan'), None
             return float('nan'), False, None, float('nan'), None

        # Filtering needed
        try:
            pdbqt_mol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=True, skip_typing=True)
            rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

            if not rdkit_mols:
                return float('nan'), False, None, float('nan'), None

            best_valid_score = float('nan')
            best_mol = None

            best_any_score = float('nan')
            best_any_mol = None

            energies = []
            if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
                energies = pdbqt_mol._pose_data["free_energies"]
            
            for idx, mol in enumerate(rdkit_mols):
                score = energies[idx] if idx < len(energies) else 999.9

                # Track global best
                if math.isnan(best_any_score) or score < best_any_score:
                    best_any_score = score
                    best_any_mol = mol

                # Check RMSD
                rmsd = self._calculate_rmsd(mol)

                if rmsd < self.rmsd_threshold:
                    if math.isnan(best_valid_score) or score < best_valid_score:
                        best_valid_score = score
                        best_mol = mol

            if not math.isnan(best_valid_score):
                return best_valid_score, True, best_mol, best_any_score, best_any_mol

            return float('nan'), False, None, best_any_score, best_any_mol

        except Exception as e:
            print(f"Error in RMSD filtering for {dlg_path}: {e}")
            return float('nan'), False, None, float('nan'), None

    def _parse_dlg_simple(self, dlg_path: Path) -> Optional[float]:
        """
        Robust parsing of best energy using Meeko's PDBQTMolecule (original logic).
        """
        try:
            pdbqt_mol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=True, skip_typing=True)
            if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
                energies = pdbqt_mol._pose_data["free_energies"]
                if energies:
                    return float(min(energies))
            
            # Fallback to manual parsing
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
            try:
                with open(dlg_path, "r") as f:
                    for line in f:
                        if "Rank | Binding Energy" in line:
                            next(f); next(f)
                            line = next(f)
                            parts = line.split("|")
                            if len(parts) >= 3:
                                return float(parts[2].strip())
            except:
                pass
            return None

    def save_best_poses_sdf(self, output_path: Union[str, Path], df_metadata: Optional[pl.DataFrame] = None, id_col: str = "id", score_col: str = "docking_score", dlg_col: str = "dlg_path"):
        """
        Extract the best pose from each successful docking run and save to an SDF.
        Adds metadata from df_metadata if provided.
        """
        if self.results_df is None:
            print("No results to save. Run score_batch first.")
            return

        print(f"Generating best poses SDF at {output_path} (using {score_col})...")
        writer = Chem.SDWriter(str(output_path))
        count = 0
        
        # Create a mapping from SMILES to metadata rows for quick lookup
        meta_map = {}
        if df_metadata is not None:
            # Check if ID column exists
            if id_col not in df_metadata.columns:
                 # Fallback to defaults if specific ID not found
                 for potential in ["molecule_chembl_id", "Name", "NAME", "compound_id"]:
                     if potential in df_metadata.columns:
                         id_col = potential
                         break
            
            # Convert to dictionary keyed by canonical_smiles
            if "canonical_smiles" in df_metadata.columns:
                key_col = "canonical_smiles"
            else:
                key_col = "smiles" # Fallback
            
            for row in df_metadata.to_dicts():
                 if row.get(key_col):
                     meta_map[row[key_col]] = row

        # Sort results by score_col (ascending = more negative = better)
        sorted_results = self.results_df.sort(score_col, descending=False)

        for row in sorted_results.to_dicts():
            if row[score_col] is None or math.isnan(row[score_col]) or row[score_col] >= 999.0 or row[dlg_col] is None:
                continue
            
            try:
                # Load DLG with Meeko
                pdbqt_mol = PDBQTMolecule.from_file(row[dlg_col], is_dlg=True, skip_typing=True)
                rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
                energies = []
                if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
                    energies = pdbqt_mol._pose_data["free_energies"]

                best_mol = None
                target_score = row[score_col]

                for idx, mol in enumerate(rdkit_mols):
                    score = energies[idx] if idx < len(energies) else 999.9

                    if abs(score - target_score) < 0.001:
                        # Found a pose with this score. 
                        # If we are using the 'valid' column, we should check RMSD if possible
                        is_valid_col = (score_col == "score_valid" or (score_col == "docking_score" and row.get("valid_pose_found", False)))
                        
                        if is_valid_col and self.reference_ligand_path and self.fragment_smiles:
                            rmsd = self._calculate_rmsd(mol)
                            if rmsd < self.rmsd_threshold:
                                best_mol = mol
                                best_mol.SetProp("RMSD_fragment", f"{rmsd:.3f}")
                                break # Found the valid one
                        else:
                            # Either we don't care about constraint (any best), or no constraint set
                            best_mol = mol
                            # Calculate RMSD for info if possible
                            if self.reference_ligand_path and self.fragment_smiles:
                                 rmsd = self._calculate_rmsd(mol)
                                 best_mol.SetProp("RMSD_fragment", f"{rmsd:.3f}")
                            break
                
                if best_mol:
                    # Add docking metadata
                    best_mol.SetProp("docking_score", str(row[score_col]))
                    best_mol.SetProp("dlg_path", str(row[dlg_col]))
                    best_mol.SetProp("n_conformers", str(row.get('n_conformers', 0)))
                    best_mol.SetProp("score_type", score_col)
                    
                    # Add external metadata (Bioactivity, ID, etc.)
                    smi = row['smiles']
                    if smi in meta_map:
                        meta = meta_map[smi]
                        # Set Name
                        if id_col in meta and meta[id_col] is not None:
                            best_mol.SetProp("_Name", str(meta[id_col]))
                        else:
                            best_mol.SetProp("_Name", str(smi))

                        # Set all other properties
                        for k, v in meta.items():
                            if v is not None and k != key_col:
                                best_mol.SetProp(str(k), str(v))
                    else:
                        best_mol.SetProp("_Name", str(smi))

                    writer.write(best_mol)
                    count += 1
            except Exception as e:
                print(f"Failed to extract pose for {row['smiles']}: {e}")

        writer.close()
        print(f"Successfully saved {count} best poses to {output_path}")
