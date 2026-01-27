# Standard library imports
import asyncio
from pathlib import Path
from typing import Optional

# Third-party imports
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr

def plot_docking_results(
    df: pl.DataFrame,
    score_col: str = "docking_score",
    activity_col: str = "standard_value", 
    valid_col: str = "valid_pose_found",
    output_path: Optional[str] = None,
    activity_units: Optional[str] = None
):
    """
    Plot docking scores vs activity values.
    
    Args:
        df: Polars DataFrame containing the results.
        score_col: Column name for docking scores.
        activity_col: Column name for activity values.
        valid_col: Column name for boolean validity (used for coloring).
        output_path: If provided, save the plot to this path.
        activity_units: Units of the activity values ('nM', 'uM', 'mM', 'M').
                       Only required if data is not pchembl_value. If None and
                       pchembl_value column exists, uses pchembl directly.
    """
    def _select_id_column() -> str:
        if "molecule_chembl_id" in df.columns:
            return "molecule_chembl_id"
        if "canonical_smiles" in df.columns:
            return "canonical_smiles"
        if "smiles" in df.columns:
            return "smiles"
        return "canonical_smiles"

    def _aggregate_best_per_id(input_df: pl.DataFrame) -> pl.DataFrame:
        id_col = _select_id_column()
        has_best_any = "score_best_any" in input_df.columns
        best_any_expr = pl.col("score_best_any") if has_best_any else pl.col(score_col)
        base_df = input_df.with_columns(
            pl.col(valid_col).fill_null(False).alias(valid_col)
        )

        grouped = base_df.group_by(id_col).agg(
            pl.col(activity_col).drop_nulls().first().alias(activity_col),
            pl.any(pl.col(valid_col) == True).alias("passed_rmsd"),
            pl.min(
                pl.when(pl.col(valid_col) == True).then(pl.col(score_col))
            ).alias("best_valid_score"),
            pl.min(best_any_expr).alias("best_any_score"),
        )

        aggregated = grouped.with_columns(
            pl.when(pl.col("passed_rmsd") == True)
            .then(pl.col("best_valid_score"))
            .otherwise(pl.col("best_any_score"))
            .alias(score_col),
            pl.col("passed_rmsd").alias(valid_col),
        ).select([id_col, activity_col, score_col, valid_col])

        return aggregated

    # Aggregate to one row per compound: enforce RMSD first, then best score
    analysis_df = _aggregate_best_per_id(df)
    print("Analysis dataframe (one row per compound):")
    print(analysis_df)

    # Filter out failed scores (999.9), nulls, and NaNs for both columns
    clean_df = analysis_df.filter(
        (pl.col(score_col).is_not_null()) &
        (pl.col(score_col).is_not_nan()) &
        (pl.col(activity_col).is_not_null()) &
        (pl.col(activity_col).is_not_nan()) &
        (pl.col(score_col) < 999.0)
    )
    
    if len(clean_df) < 2:
        print("Not enough data points to plot after filtering failed scores/NaNs.")
        return None

    # Extract columns
    scores = clean_df.get_column(score_col).to_numpy()
    activities = clean_df.get_column(activity_col).to_numpy()
    
    if valid_col in clean_df.columns:
        is_valid = clean_df.get_column(valid_col).to_numpy()
    else:
        is_valid = np.ones(len(scores), dtype=bool)

    # Check if we have pchembl_value (unit-agnostic) or need to convert
    # pchembl_value is already -log10(M), so we can use it directly
    # Note: data.py creates both pchembl_value and standard_value when pchembl is available
    # We check the original column name to determine if conversion is needed
    has_pchembl = 'pchembl_value' in df.columns
    
    if has_pchembl and activity_col == 'pchembl_value':
        # Use pchembl_value directly (already in pActivity units)
        p_activities = activities
        activity_label = "pActivity (from pchembl_value)"
    elif has_pchembl and activity_col == 'standard_value':
        # standard_value was created from pchembl_value, so it's already in pActivity units
        p_activities = activities
        activity_label = "pActivity (from pchembl_value)"
    else:
        # Convert from standard_value using units
        if activity_units is None:
            activity_units = "nM"  # Default fallback
            print(f"WARNING: No units specified, assuming {activity_units}")
        
        unit_offsets = {
            "nM": 9,
            "uM": 6,
            "mM": 3,
            "M": 0
        }
        offset = unit_offsets.get(activity_units, 9) 
        p_activities = offset - np.log10(activities + 1e-12)
        activity_label = f"pActivity (-log10 {activity_units} -> M)"

    plt.figure(figsize=(10, 6))
    
    # x=Docking Score, y=pActivity
    # Plot valid points (Blue)
    valid_mask = (is_valid == True)
    if np.any(valid_mask):
        sns.scatterplot(x=scores[valid_mask], y=p_activities[valid_mask], color='blue', alpha=0.6, label='RMSD < Threshold')

    # Plot invalid points (Red)
    invalid_mask = (is_valid == False)
    if np.any(invalid_mask):
        sns.scatterplot(x=scores[invalid_mask], y=p_activities[invalid_mask], color='red', alpha=0.6, label='RMSD > Threshold')

    def _compute_stats(x_vals: np.ndarray, y_vals: np.ndarray) -> dict:
        stats = {"n": int(len(x_vals)), "pearson": 0.0, "spearman": 0.0, "r2": 0.0}
        if len(x_vals) < 2:
            return stats
        if np.var(x_vals) > 0 and np.var(y_vals) > 0:
            pearson_corr, _ = pearsonr(x_vals, y_vals)
            spearman_corr, _ = spearmanr(x_vals, y_vals)
            stats["pearson"] = float(pearson_corr)
            stats["spearman"] = float(spearman_corr)
            stats["r2"] = float(pearson_corr ** 2)
        else:
            print("Warning: Constant input detected, correlation set to 0.0")
        return stats

    valid_stats = _compute_stats(scores[valid_mask], p_activities[valid_mask])
    all_stats = _compute_stats(scores, p_activities)
    pass_pct = 100.0 * float(valid_stats["n"]) / float(len(scores))

    stats_text = (
        f"Pass RMSD: {pass_pct:.1f}%\n"
        f"Blue (n={valid_stats['n']}): R² {valid_stats['r2']:.3f}, "
        f"Pearson {valid_stats['pearson']:.3f}, Spearman {valid_stats['spearman']:.3f}\n"
        f"All (n={all_stats['n']}): R² {all_stats['r2']:.3f}, "
        f"Pearson {all_stats['pearson']:.3f}, Spearman {all_stats['spearman']:.3f}"
    )

    plt.title("pActivity vs RMSD-Constrained Docking Score")
    plt.xlabel("Docking Score (Predicted)")
    plt.ylabel(activity_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        "score_col": score_col,
        "activity_col": activity_col,
        "activity_units": activity_units,
        "n_points": int(len(clean_df)),
        "pass_pct": float(pass_pct),
        "stats_valid": valid_stats,
        "stats_all": all_stats,
    }

def plot_activity_distribution(
    df: pl.DataFrame,
    activity_col: str = "standard_value",
    output_path: Optional[str] = None,
    activity_units: Optional[str] = None
):
    """
    Plot the distribution of bioactivity values.
    
    Args:
        df: Polars DataFrame containing activity data.
        activity_col: Column name for activity values.
        output_path: If provided, save the plot to this path.
        activity_units: Units of the activity values ('nM', 'uM', 'mM', 'M').
                       Only required if data is not pchembl_value. If None and
                       pchembl_value column exists, uses pchembl directly.
    """
    # Filter nulls
    clean_df = df.filter(pl.col(activity_col).is_not_null())
    
    if len(clean_df) == 0:
        print("No activity data to plot distribution.")
        return

    activities = clean_df.get_column(activity_col).to_numpy()
    
    # Check if we have pchembl_value (unit-agnostic) or need to convert
    has_pchembl = 'pchembl_value' in df.columns
    
    if has_pchembl and activity_col == 'pchembl_value':
        # Use pchembl_value directly (already in pActivity units)
        p_activities = activities
        activity_label = "pActivity (from pchembl_value)"
        title_suffix = "pchembl_value"
    elif has_pchembl and activity_col == 'standard_value':
        # standard_value was created from pchembl_value, so it's already in pActivity units
        p_activities = activities
        activity_label = "pActivity (from pchembl_value)"
        title_suffix = "pchembl_value"
    else:
        # Convert from standard_value using units
        if activity_units is None:
            activity_units = "nM"  # Default fallback
            print(f"WARNING: No units specified, assuming {activity_units}")
        
        unit_offsets = {"nM": 9, "uM": 6, "mM": 3, "M": 0}
        offset = unit_offsets.get(activity_units, 9)
        p_activities = offset - np.log10(activities + 1e-12)
        activity_label = f"pActivity (offset={offset})"
        title_suffix = f"Units: {activity_units}"

    plt.figure(figsize=(10, 6))
    sns.histplot(p_activities, kde=True, bins=30, color='skyblue')
    
    plt.title(f"Distribution of Experimental Activity (converted to pActivity)\n{title_suffix}, Total compounds: {len(p_activities)}")
    plt.xlabel(activity_label)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Activity distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

async def fetch_ligand_expo_sdf(
    resname: str,
    output_dir: Path,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Path]:
    """
    Fetch the ideal SDF for a ligand from RCSB Ligand Expo.

    Args:
        resname: The 3-letter ligand residue name (e.g., 'ATP').
        output_dir: Directory where the SDF file should be saved.
        session: Optional aiohttp ClientSession to reuse connections.
    """
    # Sanitize resname
    resname = resname.upper()
    url = f"https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"

    should_close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        should_close_session = True

    try:
        async with session.get(url) as response:
            if response.status == 200:
                out_path = output_dir / f"{resname}_ideal.sdf"
                text = await response.text()
                # Run file I/O in executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, out_path.write_text, text)
                return out_path
            else:
                print(f"Failed to fetch SDF for {resname} from Ligand Expo: {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching SDF for {resname}: {e}")
        return None
    finally:
        if should_close_session:
            await session.close()

def assign_bond_orders_from_template(pdb_mol: Chem.Mol, template_mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Assign bond orders to a PDB molecule using a template molecule (with bond orders).
    """
    try:
        # Remove Hs from template if the PDB mol doesn't have them
        # PDB mols from MolFromPDBFile usually don't have Hs
        if pdb_mol.GetNumAtoms() < template_mol.GetNumAtoms():
            template_mol = Chem.RemoveHs(template_mol)
            
        new_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, pdb_mol)
        return new_mol
    except Exception as e:
        print(f"Failed to assign bond orders from template: {e}")
        return None
