import multiprocessing
from pathlib import Path
from typing import Optional

from meeko import MoleculePreparation, PDBQTWriterLegacy
from molscrub import Scrub
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from .utils import effective_cpu_count

# Silence RDKit noise
RDLogger.DisableLog("rdApp.*")

# Cache heavy objects at the module level so they are initialized once per worker process
_scrubber_cache = {}
_preparator_cache = None
_writer_cache = None


class LigandPreparer:
    """Prepare SMILES molecules as PDBQT files for docking."""

    def __init__(
        self,
        n_cpus: Optional[int] = None,
        ph_low: float = 6.4,
        ph_high: float = 8.4,
        generate_isomers: bool = True,
    ):
        self.n_cpus = n_cpus or effective_cpu_count()
        self.ph_low = ph_low
        self.ph_high = ph_high
        self.generate_isomers = generate_isomers

    def _prepare_single_ligand(self, args: tuple[str, int, Path, str]) -> list[Path]:
        smiles, idx, pdbqt_dir, batch_prefix = args
        results = []

        try:
            # 1. Sanitize Input
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            # 2. Generate States (Scrub)
            global _scrubber_cache, _preparator_cache, _writer_cache
            key = (self.ph_low, self.ph_high)
            if key not in _scrubber_cache:
                _scrubber_cache[key] = Scrub(ph_low=self.ph_low, ph_high=self.ph_high)
            scrubber = _scrubber_cache[key]

            try:
                mol_states = list(scrubber(mol))
            except Exception as e:
                print(f"Scrub failed for {smiles}: {e}")
                mol_states = [mol]

            if len(mol_states) > 16:
                mol_states = mol_states[:16]

            # 3. 3D Embedding, Minimization, and Meeko Prep
            state_counter = 0

            if _preparator_cache is None:
                _preparator_cache = MoleculePreparation()
            preparator = _preparator_cache

            if _writer_cache is None:
                _writer_cache = PDBQTWriterLegacy()
            writer = _writer_cache

            for mol_state in mol_states:
                try:
                    # Add hydrogens appropriate for the specific tautomer/protonation state
                    mol_state = Chem.AddHs(mol_state)

                    params = rdDistGeom.ETKDGv3()
                    params.randomSeed = 42
                    params.useSmallRingTorsions = True

                    # Embed 3D coordinates
                    res = rdDistGeom.EmbedMultipleConfs(mol_state, numConfs=1, params=params)
                    if not res:
                        params.useRandomCoords = True
                        res = rdDistGeom.EmbedMultipleConfs(mol_state, numConfs=1, params=params)

                    if not res:
                        continue

                    # Energy Minimization using MMFF94 force field
                    try:
                        if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol_state):
                            rdForceFieldHelpers.MMFFOptimizeMolecule(
                                mol_state,
                                mmffVariant="MMFF94s",
                                maxIters=200,
                                nonBondedThresh=100.0,
                            )
                        else:
                            rdForceFieldHelpers.UFFOptimizeMolecule(mol_state, maxIters=200)
                    except Exception as e:
                        print(f"Minimization failed for {smiles} state {state_counter}: {e}")

                    # Prepare for Docking
                    mol_setups = preparator.prepare(mol_state)
                    for setup in mol_setups:
                        pdbqt_string, is_ok, error_message = writer.write_string(setup)
                        if is_ok:
                            fname = f"{batch_prefix}lig_{idx}_s{state_counter}.pdbqt"
                            fpath = pdbqt_dir / fname
                            fpath.write_text(pdbqt_string)
                            results.append(fpath)
                            state_counter += 1
                        else:
                            print(f"Meeko failed to write PDBQT for {smiles}: {error_message}")

                    if state_counter >= 32:
                        break

                except Exception as e:
                    print(f"Processing failed for a state of {smiles}: {e}")
                    continue

            return results
        except Exception as e:
            print(f"Fatal error preparing {smiles}: {e}")
            return []

    def prepare_batch(
        self, smiles_list: list[str], output_dir: Path, batch_prefix: str = ""
    ) -> list[dict]:
        """
        Prepare PDBQT files for a batch of SMILES.
        Returns a list of dictionaries: [{'smiles': str, 'pdbqt_paths': [Path]}]
        Preserves input order and duplicates.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        prep_args = [(smi, idx, output_dir, batch_prefix) for idx, smi in enumerate(smiles_list)]

        # Use spawn to avoid issues with multi-threading and fork
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=self.n_cpus) as pool:
            pdbqt_data = pool.map(self._prepare_single_ligand, prep_args)

        results = []
        for i, written_files in enumerate(pdbqt_data):
            results.append({"smiles": smiles_list[i], "pdbqt_paths": written_files})

        return results
