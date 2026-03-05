import math
from pathlib import Path
from typing import Optional, Union

import polars as pl
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem

from .utils import check_2d_match, check_validity, get_robust_match


def aggregate_results_per_id(
    df: pl.DataFrame,
    score_col: str = "docking_score",
    valid_col: str = "valid_pose_found",
    activity_col: str = "pchembl_value",
) -> pl.DataFrame:
    """
    Aggregate results to one row per compound ID.
    Enforces RMSD first, then takes best score (fallback to best_any).
    """

    def _select_id_column() -> str:
        for col in ["molecule_chembl_id", "canonical_smiles", "smiles"]:
            if col in df.columns:
                return col
        return "canonical_smiles"

    id_col = _select_id_column()

    has_best_any = "score_best_any" in df.columns
    best_any_expr = pl.col("score_best_any") if has_best_any else pl.col(score_col)

    base_df = df.with_columns(pl.col(valid_col).fill_null(False))

    agg_exprs = [
        # Only aggregate activity_col if it is NOT the ID column to avoid collision
        *(
            [pl.col(activity_col).drop_nulls().first().alias(activity_col)]
            if activity_col != id_col
            else []
        ),
        pl.col(valid_col).any().alias("passed_rmsd"),
        pl.when(pl.col(valid_col)).then(pl.col(score_col)).min().alias("best_valid_score"),
        best_any_expr.min().alias("best_any_score"),
    ]

    if "dlg_path" in base_df.columns:
        agg_exprs.extend(
            [
                pl.col("dlg_path")
                .sort_by(score_col)
                .filter(pl.col(valid_col))
                .first()
                .alias("best_valid_dlg"),
                pl.col("dlg_path").sort_by(best_any_expr).first().alias("best_any_dlg"),
            ]
        )

    grouped = base_df.group_by(id_col).agg(agg_exprs)

    derived_cols = [
        pl.when(pl.col("passed_rmsd"))
        .then(pl.col("best_valid_score"))
        .otherwise(pl.col("best_any_score"))
        .alias(score_col),
        pl.col("passed_rmsd").alias(valid_col),
        pl.col("best_valid_score").alias("score_valid"),
        pl.col("best_any_score").alias("score_best_any"),
    ]

    if "best_valid_dlg" in grouped.columns and "best_any_dlg" in grouped.columns:
        derived_cols.append(
            pl.when(pl.col("passed_rmsd"))
            .then(pl.col("best_valid_dlg"))
            .otherwise(pl.col("best_any_dlg"))
            .alias("dlg_path")
        )

    aggregated = grouped.with_columns(derived_cols)

    # Define preferred order
    desired_order = [
        id_col,
        "canonical_smiles",
        "molecule_chembl_id",
        "pchembl_value",
        activity_col,
        score_col,
        "score_valid",
        "score_best_any",
        valid_col,
        "dlg_path",
    ]

    existing_cols = [c for c in desired_order if c in aggregated.columns]
    unique_cols = list(dict.fromkeys(existing_cols))

    return aggregated.select(unique_cols)


class DockingAnalyzer:
    """Post-docking analysis: RMSD filtering, pose extraction, etc."""

    def __init__(
        self,
        reference_ligand_path: Optional[Union[str, Path]] = None,
        fragment_smiles: Optional[str] = None,
        rmsd_threshold: float = 2.0,
    ):
        self.reference_ligand_path = Path(reference_ligand_path) if reference_ligand_path else None
        self.fragment_smiles = fragment_smiles
        self.rmsd_threshold = rmsd_threshold

        self.ref_mol = None
        self.fragment_mol = None
        self.ref_match = None
        self.ref_coords = None

        if self.reference_ligand_path and self.fragment_smiles:
            self._initialize_reference()

    def _initialize_reference(self):
        """Load reference ligand and prepare fragment matching."""
        if self.reference_ligand_path.suffix.lower() == ".sdf":
            suppl = Chem.SDMolSupplier(str(self.reference_ligand_path), removeHs=False)
            self.ref_mol = next(iter(suppl), None)
        else:
            self.ref_mol = Chem.MolFromPDBFile(str(self.reference_ligand_path), removeHs=False)

        if self.ref_mol is None:
            raise ValueError(f"Could not load reference ligand from {self.reference_ligand_path}")

        self.fragment_mol = Chem.MolFromSmiles(self.fragment_smiles)
        if self.fragment_mol is None:
            self.fragment_mol = Chem.MolFromSmarts(self.fragment_smiles)
            if self.fragment_mol is None:
                raise ValueError(f"Invalid fragment SMILES/SMARTS string: {self.fragment_smiles}")

        self.ref_match = get_robust_match(self.ref_mol, self.fragment_mol)
        if not self.ref_match:
            print(
                f"WARNING: Reference ligand ({self.reference_ligand_path.name}) does not match fragment SMILES!"
            )
        else:
            ref_conf = self.ref_mol.GetConformer()
            self.ref_coords = []
            for idx in self.ref_match:
                pos = ref_conf.GetAtomPosition(idx)
                self.ref_coords.append((pos.x, pos.y, pos.z))

    def calculate_rmsd(self, probe_mol: Chem.Mol, conf_id: int = -1) -> float:
        """Calculate RMSD of the fragment between probe_mol and self.ref_mol."""
        if self.ref_mol is None or self.fragment_mol is None or not self.ref_match:
            return 0.0

        probe_match = get_robust_match(probe_mol, self.fragment_mol)
        if not probe_match:
            return 999.9

        try:
            probe_conf = probe_mol.GetConformer(conf_id)
        except ValueError:
            return 999.9

        probe_coords = []
        for idx in probe_match:
            pos = probe_conf.GetAtomPosition(idx)
            probe_coords.append((pos.x, pos.y, pos.z))

        sq_diff = 0
        for (rx, ry, rz), (px, py, pz) in zip(self.ref_coords, probe_coords):
            sq_diff += (rx - px) ** 2 + (ry - py) ** 2 + (rz - pz) ** 2

        return math.sqrt(sq_diff / len(self.ref_coords))

    def filter_poses_by_rmsd(
        self, pose_file: Path, smiles: str
    ) -> tuple[float, bool, Optional[Chem.Mol], float, Optional[Chem.Mol], int, int]:
        """
        Parse DLG or PDBQT, filter poses by RMSD if applicable.

        Returns:
            (best_valid_score, passed_constraint, best_valid_mol,
             best_any_score, best_any_mol,
             best_valid_pose_index, best_any_pose_index)

        pose_index values are 0-based indices into the list returned by
        RDKitMolCreate.from_pdbqt_mol (after Vina multi-conformer unrolling).
        -1 indicates no pose was found.
        """
        try:
            pose_file = Path(pose_file)
            is_dlg = pose_file.suffix.lower() == ".dlg"
            pdbqt_mol = PDBQTMolecule.from_file(str(pose_file), is_dlg=is_dlg, skip_typing=True)
            rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

            if not rdkit_mols:
                return float("nan"), False, None, float("nan"), None, -1, -1

            # Unroll multi-conformer mols (typical for Vina PDBQT output)
            if len(rdkit_mols) == 1 and rdkit_mols[0].GetNumConformers() > 1:
                base_mol = rdkit_mols[0]
                unrolled = []
                for conf in base_mol.GetConformers():
                    new_mol = Chem.Mol(base_mol)
                    new_mol.RemoveAllConformers()
                    new_mol.AddConformer(conf, assignId=True)
                    unrolled.append(new_mol)
                rdkit_mols = unrolled

            best_valid_score = float("nan")
            best_valid_mol = None
            best_valid_idx = -1
            best_any_score = float("nan")
            best_any_mol = None
            best_any_idx = -1

            energies = getattr(pdbqt_mol, "_pose_data", {}).get("free_energies", [])
            energies_len = len(energies)

            for idx, mol in enumerate(rdkit_mols):
                score = energies[idx] if idx < energies_len else 999.9

                if math.isnan(best_any_score) or score < best_any_score:
                    best_any_score = score
                    best_any_mol = mol
                    best_any_idx = idx

                rmsd = self.calculate_rmsd(mol)
                if rmsd < self.rmsd_threshold:
                    if math.isnan(best_valid_score) or score < best_valid_score:
                        best_valid_score = score
                        best_valid_mol = mol
                        best_valid_idx = idx

            passed = not math.isnan(best_valid_score)
            return (
                best_valid_score if passed else float("nan"),
                passed,
                best_valid_mol,
                best_any_score,
                best_any_mol,
                best_valid_idx,
                best_any_idx,
            )

        except Exception as e:
            print(f"Error in RMSD filtering for {pose_file}: {e}")
            return float("nan"), False, None, float("nan"), None, -1, -1

    def save_best_poses_sdf(
        self,
        output_path: Union[str, Path],
        results_df: pl.DataFrame,
        df_metadata: Optional[pl.DataFrame] = None,
        id_col: str = "id",
        score_col: str = "docking_score",
        dlg_col: str = "dlg_path",
    ):
        """
        Extract the best pose from each successful docking run and save to an SDF.
        Adds metadata from df_metadata if provided.

        Pose selection strategy:
          1. Parse all poses from the DLG/PDBQT file.
          2. If RMSD checking is available, prefer the lowest-energy pose that
             passes the fragment-RMSD threshold.
          3. Fall back to the best-energy pose regardless of RMSD, so the SDF
             always contains an entry for every molecule in the top-N list.
        """
        print(f"Generating best poses SDF at {output_path} (using {score_col})...")
        writer = Chem.SDWriter(str(output_path))
        count = 0

        # Build metadata lookup keyed by SMILES
        meta_map = {}
        if df_metadata is not None:
            if id_col not in df_metadata.columns:
                for potential in ["molecule_chembl_id", "Name", "NAME", "compound_id"]:
                    if potential in df_metadata.columns:
                        id_col = potential
                        break
            key_col = "canonical_smiles" if "canonical_smiles" in df_metadata.columns else "smiles"
            for row in df_metadata.to_dicts():
                if row.get(key_col):
                    meta_map[row[key_col]] = row

        # Filter to rows that have a valid score and a DLG path; sort best-first
        filtered_results = results_df.filter(
            pl.col(score_col).is_not_null()
            & pl.col(score_col).is_not_nan()
            & (pl.col(score_col) < 999.0)
            & pl.col(dlg_col).is_not_null()
        ).sort(score_col, descending=False)

        use_rmsd = self.ref_mol is not None and self.fragment_mol is not None
        check_rmsd_for_row = score_col in ("score_valid", "docking_score")

        for row in filtered_results.iter_rows(named=True):
            try:
                pose_file = Path(row[dlg_col])
                if not pose_file.exists():
                    print(f"  [skip] DLG not found: {pose_file}")
                    continue

                is_dlg = pose_file.suffix.lower() == ".dlg"
                pdbqt_mol = PDBQTMolecule.from_file(str(pose_file), is_dlg=is_dlg, skip_typing=True)
                rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

                if not rdkit_mols:
                    print(f"  [skip] No RDKit molecules from {pose_file.name}")
                    continue

                # Collect (energy, mol) pairs; energies come from pose_data if available
                energies = []
                if hasattr(pdbqt_mol, "_pose_data") and "free_energies" in pdbqt_mol._pose_data:
                    energies = list(pdbqt_mol._pose_data["free_energies"])

                pose_pairs: list[tuple[float, Chem.Mol]] = []
                for idx, mol in enumerate(rdkit_mols):
                    energy = energies[idx] if idx < len(energies) else 999.9
                    pose_pairs.append((energy, mol))

                # Sort by energy ascending (best = lowest free energy)
                pose_pairs.sort(key=lambda t: t[0])

                chosen_mol = None
                chosen_rmsd: Optional[float] = None

                if use_rmsd and check_rmsd_for_row:
                    # Prefer the lowest-energy pose that satisfies the RMSD threshold
                    for energy, mol in pose_pairs:
                        rmsd = self.calculate_rmsd(mol)
                        if rmsd < self.rmsd_threshold:
                            chosen_mol = mol
                            chosen_rmsd = rmsd
                            break

                # Fall back to best-energy pose if no RMSD-passing pose found
                if chosen_mol is None:
                    _, mol = pose_pairs[0]
                    chosen_mol = mol
                    if use_rmsd:
                        chosen_rmsd = self.calculate_rmsd(mol)

                # Annotate and write
                smi = row["smiles"]
                chosen_mol.SetProp("SMILES", smi)
                chosen_mol.SetProp("docking_score", str(row[score_col]))
                chosen_mol.SetProp("normalized_score", str(row.get("normalized_score", "")))
                chosen_mol.SetProp("dlg_path", str(row[dlg_col]))
                chosen_mol.SetProp("score_type", score_col)
                if chosen_rmsd is not None:
                    chosen_mol.SetProp("RMSD_fragment", f"{chosen_rmsd:.3f}")
                    chosen_mol.SetProp(
                        "rmsd_passed_threshold",
                        "true" if chosen_rmsd < self.rmsd_threshold else "false",
                    )

                if smi in meta_map:
                    meta = meta_map[smi]
                    chosen_mol.SetProp("_Name", str(meta.get(id_col, smi)))
                    for k, v in meta.items():
                        if v is not None:
                            chosen_mol.SetProp(str(k), str(v))
                else:
                    chosen_mol.SetProp("_Name", smi)

                writer.write(chosen_mol)
                count += 1

            except Exception as e:
                print(f"  [error] Failed to extract pose for {row.get('smiles', '?')}: {e}")

        writer.close()
        print(f"Successfully saved {count} best poses to {output_path}")
