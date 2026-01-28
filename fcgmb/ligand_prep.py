import multiprocessing
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom
from meeko import MoleculePreparation, PDBQTWriterLegacy
from molscrub import Scrub

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')

class LigandPreparer:
    """Prepare SMILES molecules as PDBQT files for docking."""
    
    def __init__(
        self, 
        n_cpus: Optional[int] = None, 
        ph_low: float = 6.4, 
        ph_high: float = 8.4, 
        generate_isomers: bool = True
    ):
        self.n_cpus = n_cpus or multiprocessing.cpu_count()
        self.ph_low = ph_low
        self.ph_high = ph_high
        self.generate_isomers = generate_isomers

    def _prepare_single_ligand(self, args: Tuple[str, int, Path, str]) -> List[Path]:
        """
        Prepare multiple ligand PDBQTs (states/conformers) for a single SMILES using Scrub and Meeko.
        Returns a list of written file paths.
        """
        smiles, idx, pdbqt_dir, batch_prefix = args
        results = []
        
        try:
            # 1. Sanitize Input
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            # 2. Analyze Stereo
            input_iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            has_stereo = bool(centers)

            # 3. Generate States (Scrub)
            scrubber = Scrub(ph_low=self.ph_low, ph_high=self.ph_high)
            mol_states = []

            try:
                scrub_results = list(scrubber(mol))
                if (not self.generate_isomers) and has_stereo:
                    filtered_states = []
                    for s_mol in scrub_results:
                        s_iso = Chem.MolToSmiles(s_mol, isomericSmiles=True, canonical=True)
                        if input_iso_smiles == s_iso:
                            filtered_states.append(s_mol)
                    
                    mol_states = filtered_states if filtered_states else [mol]
                else:
                    mol_states = scrub_results
            except Exception as e:
                print(f"Scrub failed for {smiles}: {e}")
                mol_states = [mol]

            if len(mol_states) > 16:
                mol_states = mol_states[:16]

            # 4. 3D Embedding and Meeko Prep
            state_counter = 0
            preparator = MoleculePreparation()

            for mol_state in mol_states:
                try:
                    mol_state = Chem.AddHs(mol_state)
                    params = rdDistGeom.ETKDGv3()
                    params.randomSeed = 42
                    params.useSmallRingTorsions = True
                    
                    res = rdDistGeom.EmbedMultipleConfs(mol_state, numConfs=1, params=params)
                    if not res:
                        params.useRandomCoords = True
                        res = rdDistGeom.EmbedMultipleConfs(mol_state, numConfs=1, params=params)
                        
                    if not res:
                        continue

                    mol_setups = preparator.prepare(mol_state)
                    for setup in mol_setups:
                        pdbqt_string, is_ok, error_message = PDBQTWriterLegacy().write_string(setup)
                        if is_ok:
                            fname = f"{batch_prefix}lig_{idx}_s{state_counter}.pdbqt"
                            fpath = pdbqt_dir / fname
                            fpath.write_text(pdbqt_string)
                            results.append(fpath)
                            state_counter += 1
                    
                    if state_counter >= 32:
                        break
                except Exception:
                    continue
                    
            return results
        except Exception:
            return []

    def prepare_batch(self, smiles_list: List[str], output_dir: Path, batch_prefix: str = "") -> Dict[str, List[Path]]:
        """
        Prepare PDBQT files for a batch of SMILES. 
        Returns {smiles: [pdbqt_paths]}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        prep_args = [(smi, idx, output_dir, batch_prefix) for idx, smi in enumerate(smiles_list)]
        
        # Use spawn to avoid issues with multi-threading and fork
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=self.n_cpus) as pool:
            pdbqt_data = pool.map(self._prepare_single_ligand, prep_args)
        
        smiles_to_paths = {}
        for i, written_files in enumerate(pdbqt_data):
            smi = smiles_list[i]
            smiles_to_paths[smi] = written_files
            
        return smiles_to_paths
