import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
exps_dir = Path("/work/users/s/h/shuhang/benchmark/exps")
exps_ub_dir = Path("/work/users/s/h/shuhang/benchmark/exps_upperbound")
benchmark_dir = Path("/work/users/s/h/shuhang/benchmark")

MODEL_RENAME_MAP = {
    "acegen-a2c": "A2C",
    "acegen-ahc": "AHC",
    "acegen-ppo": "PPO",
    "acegen-ppod": "PPOD",
    "acegen-reinforce": "REINFORCE",
    "acegen-reinvent": "REINVENT",
    "libinvent": "LibINVENT",
    "genmol": "GenMol",
    "invirtuogen": "InVirtuoGen",
}

def load_data_for_dir(base_dir, exp_type):
    data_rows = []
    # Find all results.csv
    csv_paths = glob.glob(str(base_dir / "*/run_*/*/results.csv"))
    print(f"Found {len(csv_paths)} results.csv files in {base_dir}")
    
    for path_str in csv_paths:
        p = Path(path_str)
        # Directory structure: base_dir / model / run_id / target / results.csv
        target = p.parent.name
        run_id = p.parent.parent.name
        model_raw = p.parent.parent.parent.name
        model = MODEL_RENAME_MAP.get(model_raw, model_raw)
        
        # Load results.csv
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue
            
        # Correct the dlg_path column to point to local files if they exist under p.parent / "poses"
        if 'dlg_path' in df.columns:
            poses_dir = p.parent / "poses"
            local_poses = set()
            if poses_dir.exists():
                try:
                    local_poses = set(os.listdir(poses_dir))
                except Exception:
                    pass
            
            def correct_path(val):
                if pd.isna(val):
                    return val
                filename = Path(val).name
                if filename in local_poses:
                    return str(poses_dir / filename)
                return val
            df['dlg_path'] = df['dlg_path'].apply(correct_path)
            
        # Filter valid poses
        df_valid = df[df['valid_pose_found'] == True].copy()
        if df_valid.empty:
            continue
            
        # Add basic columns
        df_valid['exp_type'] = exp_type
        df_valid['model'] = model
        df_valid['run_id'] = run_id
        df_valid['target'] = target
        
        # Merge metrics cache
        cache_p = p.parent / "molecule_metrics_cache.csv"
        if cache_p.exists():
            try:
                cache_df = pd.read_csv(cache_p)
                df_valid = df_valid.merge(cache_df[['smiles', 'qed', 'sa']], on='smiles', how='left')
            except Exception as e:
                print(f"Error merging metrics cache in {p.parent}: {e}")
                
        # Merge molskill
        ms_p = p.parent / "scores_molskill.csv"
        if ms_p.exists():
            try:
                ms_df = pd.read_csv(ms_p)
                df_valid = df_valid.merge(ms_df[['smiles', 'molskill_score']], on='smiles', how='left')
            except Exception as e:
                print(f"Error merging molskill scores in {p.parent}: {e}")
                
        # Merge stoplight
        sl_p = p.parent / "scores_stoplight.csv"
        if sl_p.exists():
            try:
                sl_df = pd.read_csv(sl_p)
                df_valid = df_valid.merge(sl_df[['smiles', 'stoplight_score']], on='smiles', how='left')
            except Exception as e:
                print(f"Error merging stoplight scores in {p.parent}: {e}")
                
        # Merge aizynthfinder
        az_p = p.parent / "scores_aizynthfinder.csv"
        if az_p.exists():
            try:
                az_df = pd.read_csv(az_p)
                df_valid = df_valid.merge(az_df[['smiles', 'aizynthfinder_state_score']], on='smiles', how='left')
            except Exception as e:
                print(f"Error merging aizynthfinder scores in {p.parent}: {e}")
                
        data_rows.append(df_valid)
        
    if not data_rows:
        return pd.DataFrame()
        
    return pd.concat(data_rows, ignore_index=True)

print("Loading Uncapped experiments...")
df_uncapped = load_data_for_dir(exps_dir, "Uncapped")
print("Loading Capped experiments...")
df_capped = load_data_for_dir(exps_ub_dir, "Capped")

# Combine all
all_df = pd.concat([df_uncapped, df_capped], ignore_index=True)
if all_df.empty:
    print("No valid pose data found!")
    exit(1)

# Ensure numeric types
metrics = ['docking_score', 'norm_score', 'qed', 'sa', 'molskill_score', 'stoplight_score', 'aizynthfinder_state_score']
for col in metrics:
    all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

# Apply updated normalization specifications
print("Normalizing metrics with strict bounds...")

# 1. Docking Score component: norm_score directly (if NaN, fall back to target min-max score)
# Note: Uncapped experiments are allowed to exceed 1.0, so we only clip the lower bound to 0.0
def fill_missing_norm_score(group):
    d = group['docking_score']
    d_min, d_max = d.min(), d.max()
    denom = d_max - d_min + 1e-8
    fallback = (d_max - d) / denom if d_max != d_min else pd.Series(0.5, index=group.index)
    group['norm_score'] = group['norm_score'].fillna(fallback)
    return group

all_df = all_df.groupby('target', group_keys=False).apply(fill_missing_norm_score)
all_df['norm_docking'] = np.maximum(all_df['norm_score'], 0.0)

# 2. QED: 0-1 (higher is better)
all_df['norm_qed'] = np.clip(all_df['qed'], 0.0, 1.0)

# 3. SA: 1-10 (lower is better) -> (10 - SA) / 9
all_df['norm_sa'] = np.clip((10.0 - all_df['sa']) / 9.0, 0.0, 1.0)

# 4. MolSkill: -30 to 40 (lower/more negative is better) -> (40 - MolSkill) / 70
all_df['norm_molskill'] = np.clip((40.0 - all_df['molskill_score']) / 70.0, 0.0, 1.0)

# 5. STOPLIGHT: 0 to 2 (lower is better) -> (2 - STOPLIGHT) / 2
all_df['norm_stoplight'] = np.clip((2.0 - all_df['stoplight_score']) / 2.0, 0.0, 1.0)

# 6. AIZynthFinder: 0 to 1 (higher is better)
all_df['norm_aizynth'] = np.clip(all_df['aizynthfinder_state_score'], 0.0, 1.0)

# Compute Geometric Mean
# Log-average method with a clamp at 1e-4 to avoid log(0) = -inf
epsilon = 1e-4
norm_cols = ['norm_docking', 'norm_qed', 'norm_sa', 'norm_molskill', 'norm_stoplight', 'norm_aizynth']

def calc_geometric_mean(row):
    vals = [row[c] for c in norm_cols if not pd.isna(row[c])]
    if not vals:
        return np.nan
    clamped_vals = np.clip(vals, epsilon, None) # Allow values to exceed 1.0
    return np.exp(np.mean(np.log(clamped_vals)))

print("Computing geometric mean scores...")
all_df['cumulative_score'] = all_df.apply(calc_geometric_mean, axis=1)

# Now, find the top pose (highest geometric mean score) for each (exp_type, model, target)
idx_top = all_df.groupby(['exp_type', 'model', 'target'])['cumulative_score'].idxmax()
top_df = all_df.loc[idx_top].copy()

# Sort top_df
top_df = top_df.sort_values(by=['model', 'exp_type', 'target'])

# Create relative paths for dlg_path (relative to benchmark_dir)
def get_relative_path(p_str):
    try:
        p = Path(p_str)
        return str(p.relative_to(benchmark_dir))
    except Exception:
        if 'exps/' in p_str:
            return 'exps/' + p_str.split('exps/')[-1]
        elif 'exps_upperbound/' in p_str:
            return 'exps_upperbound/' + p_str.split('exps_upperbound/')[-1]
        return p_str

top_df['rel_pose_path'] = top_df['dlg_path'].apply(get_relative_path)

print(f"Extracted {len(top_df)} top poses.")

# Generate Markdown table separated by model
md_content = """# Top Showcase Docking Poses (Geometric Mean Selection)

This document lists the best-performing molecular docking poses generated across all benchmark targets and generative models. The candidates are selected using a **Cumulative Score** calculated as the **Geometric Mean** of six normalized metrics:
1. **Docking Score** (AutoDock / Gnina binding energy; higher is better when normalized. Already normalized in results as `norm_score`)
2. **QED** (Quantitative Estimate of Drug-likeness; range $[0, 1]$, higher is better)
3. **SA** (Synthetic Accessibility; range $[1, 10]$, lower is better, normalized using $\\frac{10 - SA}{9}$)
4. **MolSkill Score** (Chemist organic synthetic preference; range $[-30, 40]$, lower is better, normalized using $\\frac{40 - MolSkill}{70}$)
5. **STOPLIGHT Score** (ADMET toxicity risk; range $[0, 2]$, lower is better, normalized using $\\frac{2 - STOPLIGHT}{2}$)
6. **AIZynthFinder State Score** (Route feasibility; range $[0, 1]$, higher is better)

### Normalization Bounds & Scaling
| Metric | Original Range / Condition | Best Value | Normalization Mapping |
| :--- | :--- | :--- | :--- |
| **Docking Score** | Already scaled as `norm_score` | High | Direct value (Clipped to $\\ge 0$, uncapped above $1.0$) |
| **QED** | $[0, 1]$ | $1.0$ | Direct value (Clipped to $[0, 1]$) |
| **SA** | $[1, 10]$ | $1.0$ | $\\frac{10 - SA}{9}$ (Clipped to $[0, 1]$) |
| **MolSkill** | Unbounded (Scaled $[-30, 40]$) | $-30.0$ | $\\frac{40 - MolSkill}{70}$ (Clipped to $[0, 1]$) |
| **STOPLIGHT**| $[0, 2]$ | $0.0$ | $\\frac{2 - STOPLIGHT}{2}$ (Clipped to $[0, 1]$) |
| **AIZynthFinder**| $[0, 1]$ | $1.0$ | Direct state score (Clipped to $[0, 1]$) |

*The geometric mean is calculated over all available normalized metrics. For stability, normalized values are clamped to a minimum of $10^{-4}$ prior to calculating the geometric mean. Note that in Uncapped runs, the Docking Score component can exceed $1.0$ (signifying optimization exceeding the initial benchmark target threshold), allowing both the Normalized Docking and the Geometric Mean Cumulative Score to go above $1.0$.*

---
"""

# Unique models in the sorted dataset
models = top_df['model'].unique()

for model in models:
    model_df = top_df[top_df['model'] == model]
    
    md_content += f"\n## Model: {model}\n\n"
    md_content += """| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
"""
    
    for _, row in model_df.iterrows():
        cum_score = f"{row['cumulative_score']:.3f}"
        dock_raw = f"{row['docking_score']:.2f}" if not pd.isna(row['docking_score']) else "N/A"
        dock_norm = f"{row['norm_docking']:.3f}" if not pd.isna(row['norm_docking']) else "N/A"
        qed = f"{row['qed']:.3f}" if not pd.isna(row['qed']) else "N/A"
        sa = f"{row['sa']:.2f}" if not pd.isna(row['sa']) else "N/A"
        molskill = f"{row['molskill_score']:.2f}" if not pd.isna(row['molskill_score']) else "N/A"
        stoplight = f"{row['stoplight_score']:.2f}" if not pd.isna(row['stoplight_score']) else "N/A"
        aizynth = f"{row['aizynthfinder_state_score']:.3f}" if not pd.isna(row['aizynthfinder_state_score']) else "N/A"
        pose_idx = int(row['pose_index']) if not pd.isna(row['pose_index']) else "N/A"
        
        md_content += f"| {row['exp_type']} | {row['target']} | **{cum_score}** | {dock_raw} | {dock_norm} | {qed} | {sa} | {molskill} | {stoplight} | {aizynth} | `{row['rel_pose_path']}` | {pose_idx} |\n"

md_content += """
## How to Retrieve the Poses

To grab any specific pose for visualization (e.g., in PyMOL or ChimeraX), locate the `.dlg` file listed in the table. 
The `.dlg` file contains the docking run outputs (in AutoDock format). The `Pose Index` indicates the zero-indexed conformation model number within the file.

### DLG to SDF Conversion Utility
You can also use the conversion script provided in `scripts/analysis/convert_dlg_to_sdf.py` to extract any conformation pose to a `.sdf` file:

```bash
python3 scripts/analysis/convert_dlg_to_sdf.py \\
    -i <path_to_dlg_file> \\
    -o <output_path_to_sdf> \\
    -p <pose_index>
```
"""

output_path = benchmark_dir / "top_showcase_poses.md"
with open(output_path, "w") as f:
    f.write(md_content)
print(f"Successfully wrote top poses table to {output_path}")
