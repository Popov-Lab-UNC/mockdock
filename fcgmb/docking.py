from abc import ABC, abstractmethod
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import polars as pl
from rdkit import RDLogger

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')

class DockingOracle(ABC):
    """Abstract base class for docking oracles."""
    
    def __init__(
        self, 
        receptor_file: Union[str, Path], 
        n_poses: int = 10, 
        n_cpus: Optional[int] = None, 
        n_gpus: Optional[int] = None, 
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        self.receptor_file = Path(receptor_file).resolve()
        self.n_poses = n_poses
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.n_gpus = n_gpus if n_gpus is not None else 1
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if not self.receptor_file.exists():
            raise FileNotFoundError(f"Receptor file not found: {self.receptor_file}")

    @abstractmethod
    def dock_batch(self, smiles_to_pdbqts: Dict[str, List[Path]], chunk_idx: int) -> List[Dict]:
        """Dock a batch of prepared ligand PDBQT files. Returns list of docking results."""
        pass

class AutoDockGPUOracle(DockingOracle):
    # Extensions required for AutoDock-GPU docking
    ALLOWED_RECEPTOR_EXTENSIONS = {".map", ".fld", ".xyz", ".pdbqt"}

    def __init__(
        self,
        receptor_file: Union[str, Path],
        adgpu_executable: str = "adgpu",
        n_poses: int = 10,
        n_cpus: Optional[int] = None,
        n_gpus: int = 1,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(receptor_file, n_poses, n_cpus, n_gpus, save_dir)
        
        self.adgpu_executable = adgpu_executable
        
        # Check if adgpu is callable
        resolved_exe = shutil.which(self.adgpu_executable)
        if resolved_exe is None:
            if Path(self.adgpu_executable).exists() and os.access(self.adgpu_executable, os.X_OK):
                resolved_exe = str(Path(self.adgpu_executable).resolve())
            else:
                raise FileNotFoundError(f"AutoDock-GPU executable not found or not executable: {self.adgpu_executable}")
        self.adgpu_executable = resolved_exe

    def dock_batch(self, smiles_to_pdbqts: Dict[str, List[Path]], chunk_idx: int) -> List[Dict]:
        """Implementation of ADGPU docking for a batch of PDBQT files."""
        with tempfile.TemporaryDirectory(prefix=f"adgpu_chunk_{chunk_idx}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Map from index to (smiles, rel_path, stem) for parsing later
            idx_to_info = []
            
            # Create filelist and prepare files in tmp_dir
            filelist_path = tmp_path / "filelist.txt"
            
            # Symlink receptor files
            receptor_dir = self.receptor_file.parent
            for ref_file in receptor_dir.iterdir():
                if ref_file.is_file() and (
                    ref_file.suffix in self.ALLOWED_RECEPTOR_EXTENSIONS or
                    ref_file.name == self.receptor_file.name
                ):
                    target = tmp_path / ref_file.name
                    if not target.exists():
                        try:
                            os.symlink(ref_file.resolve(), target)
                        except OSError:
                            shutil.copy2(ref_file, target)
            
            # Write filelist and track info
            with open(filelist_path, "w") as f:
                f.write(f"{self.receptor_file.name}\n")

                # Create ligands directory once
                lig_dir = tmp_path / "ligands"
                lig_dir.mkdir(exist_ok=True)

                for smi, paths in smiles_to_pdbqts.items():
                    for p in paths:
                        # Copy PDBQT to tmp_path/ligands
                        target_p = lig_dir / p.name
                        shutil.copy2(p, target_p)
                        
                        rel_path = f"ligands/{p.name}"
                        stem = p.stem
                        f.write(f"{rel_path}\n")
                        f.write(f"{stem}\n")
                        idx_to_info.append({"smiles": smi, "stem": stem, "pdbqt_path": p})

            if not idx_to_info:
                return []

            # Run ADGPU
            env = os.environ.copy()
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
                subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, env=env, text=True)
            except Exception as e:
                print(f"Exception running ADGPU: {e}")

            # Collect DLG paths
            final_results = []
            for info in idx_to_info:
                dlg_path = tmp_path / f"{info['stem']}.dlg"
                if dlg_path.exists():
                    persistent_dlg = None
                    if self.save_dir:
                        persistent_dlg = self.save_dir / f"chunk_{chunk_idx}_{info['stem']}.dlg"
                        shutil.copy2(dlg_path, persistent_dlg)
                    
                    final_results.append({
                        "smiles": info["smiles"],
                        "dlg_path": persistent_dlg or dlg_path,
                        "pdbqt_path": info["pdbqt_path"]
                    })
                else:
                    final_results.append({
                        "smiles": info["smiles"],
                        "dlg_path": None,
                        "pdbqt_path": info["pdbqt_path"]
                    })
            
            return final_results

class AutoDockVinaOracle(DockingOracle):
    def __init__(
        self,
        receptor_file: Union[str, Path],
        n_poses: int = 10,
        n_cpus: Optional[int] = None,
        exhaustiveness: int = 32,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(receptor_file, n_poses, n_cpus, n_gpus=0, save_dir=save_dir)
        self.exhaustiveness = exhaustiveness
        
        from vina import Vina
        # Use AD4 scoring function since we are using AutoGrid maps
        self.v = Vina(sf_name='ad4', cpu=self.n_cpus, verbosity=0)
        
        # Load maps
        map_prefix = str(self.receptor_file).replace('.maps.fld', '')
        self.v.load_maps(map_prefix)

    def dock_batch(self, smiles_to_pdbqts: Dict[str, List[Path]], chunk_idx: int) -> List[Dict]:
        """Implementation of Vina docking for a batch of PDBQT files."""
        final_results = []
        
        for smiles, paths in smiles_to_pdbqts.items():
            for p in paths:
                try:
                    self.v.set_ligand_from_file(str(p))
                    self.v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
                    
                    stem = p.stem
                    output_pdbqt = p.parent / f"{stem}_docked.pdbqt"
                    self.v.write_poses(str(output_pdbqt), n_poses=self.n_poses, overwrite=True)
                    
                    persistent_pdbqt = None
                    if self.save_dir:
                        persistent_pdbqt = self.save_dir / f"chunk_{chunk_idx}_{stem}_docked.pdbqt"
                        shutil.copy2(output_pdbqt, persistent_pdbqt)
                    
                    final_results.append({
                        "smiles": smiles,
                        "dlg_path": persistent_pdbqt or output_pdbqt, # Reusing dlg_path key for convenience in pipeline
                        "pdbqt_path": p
                    })
                except Exception as e:
                    print(f"Exception running Vina for {smiles}: {e}")
                    final_results.append({
                        "smiles": smiles,
                        "dlg_path": None,
                        "pdbqt_path": p
                    })
                    
        return final_results
