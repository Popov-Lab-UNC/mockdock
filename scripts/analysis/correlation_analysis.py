import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor

# Configuration
exps_dir = Path("/work/users/s/h/shuhang/benchmark/exps")
exps_ub_dir = Path("/work/users/s/h/shuhang/benchmark/exps_upperbound")
bioactivity_dir = Path("/work/users/s/h/shuhang/benchmark/src/mockdock/bioactivity_data")
assets_dir = Path("/work/users/s/h/shuhang/benchmark/assets/correlation")
output_md_path = Path("/work/users/s/h/shuhang/benchmark/correlation_analysis.md")

# Strict order requested by the user
MODEL_ORDER = ["A2C", "AHC", "PPO", "PPOD", "REINFORCE", "REINVENT", "LibINVENT", "GenMol", "InVirtuoGen"]
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

# Parallel worker function
def calc_descriptors_chunk(smiles_list):
    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                results.append((smi, float(mw), float(logp)))
                continue
        except Exception:
            pass
        results.append((smi, np.nan, np.nan))
    return results

def main():
    # 1. Load Reference Data
    print("Loading Reference datasets...")
    ref_data_list = []
    for path in bioactivity_dir.glob("*.csv"):
        target = path.stem
        try:
            df = pd.read_csv(path)
            smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
            
            cols_to_use = [smi_col]
            if "norm_score" in df.columns:
                cols_to_use.append("norm_score")
                
            df_clean = df[cols_to_use].dropna(subset=[smi_col]).copy()
            df_clean = df_clean.rename(columns={smi_col: "smiles"})
            if "norm_score" not in df_clean.columns:
                df_clean["norm_score"] = np.nan
                
            df_clean["target"] = target
            df_clean["source"] = "Reference"
            df_clean["model"] = "Reference"
            ref_data_list.append(df_clean)
        except Exception as e:
            print(f"Error loading reference {target}: {e}")

    ref_df = pd.concat(ref_data_list, ignore_index=True) if ref_data_list else pd.DataFrame()
    print(f"Loaded {len(ref_df)} reference molecules.")

    # 2. Load Generated Data (> 0.0 norm_score)
    def load_generated_data(base_dir, exp_label):
        data_list = []
        csv_paths = glob.glob(str(base_dir / "*/run_*/*/results.csv"))
        print(f"Found {len(csv_paths)} results files in {base_dir}")
        
        for path_str in csv_paths:
            p = Path(path_str)
            target = p.parent.name
            model_raw = p.parent.parent.parent.name
            model = MODEL_RENAME_MAP.get(model_raw, model_raw)
            
            try:
                df = pd.read_csv(p)
                if "norm_score" in df.columns and "smiles" in df.columns:
                    df_filtered = df[(df["norm_score"] > 0.0) & (df["valid_pose_found"] == True)].copy()
                    if not df_filtered.empty:
                        df_filtered = df_filtered.drop_duplicates(subset=["smiles"]).copy()
                        df_sub = df_filtered[["smiles", "norm_score"]].copy()
                        df_sub["target"] = target
                        df_sub["model"] = model
                        df_sub["source"] = f"Generated ({exp_label})"
                        df_sub["exp_type"] = exp_label
                        data_list.append(df_sub)
            except Exception as e:
                print(f"Error parsing {p}: {e}")
                
        if not data_list:
            return pd.DataFrame()
        return pd.concat(data_list, ignore_index=True)

    print("Loading Uncapped generated molecules...")
    gen_uncapped = load_generated_data(exps_dir, "Uncapped")
    print("Loading Capped generated molecules...")
    gen_capped = load_generated_data(exps_ub_dir, "Capped")

    # Combine all generated
    gen_df = pd.concat([gen_uncapped, gen_capped], ignore_index=True)
    if gen_df.empty:
        print("No generated molecules found!")
        exit(1)
        
    print(f"Loaded {len(gen_df)} generated molecules with norm_score > 0.0.")

    # 3. Calculate descriptors in parallel
    all_smiles = pd.concat([ref_df["smiles"], gen_df["smiles"]]).unique()
    print(f"Found {len(all_smiles)} total unique SMILES. Processing in parallel...")
    
    n_workers = min(os.cpu_count() or 4, 16)
    chunk_size = int(np.ceil(len(all_smiles) / (n_workers * 4)))
    chunks = [all_smiles[i:i + chunk_size] for i in range(0, len(all_smiles), chunk_size)]
    
    desc_dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calc_descriptors_chunk, chunk) for chunk in chunks]
        for f in futures:
            for smi, mw, logp in f.result():
                desc_dict[smi] = (mw, logp)
                
    # Map back to Reference set
    ref_df["mw"] = ref_df["smiles"].map(lambda x: desc_dict.get(x, (np.nan, np.nan))[0])
    ref_df["logp"] = ref_df["smiles"].map(lambda x: desc_dict.get(x, (np.nan, np.nan))[1])
    ref_df = ref_df.dropna(subset=["mw", "logp"])
    
    # Map back to Generated set
    gen_df["mw"] = gen_df["smiles"].map(lambda x: desc_dict.get(x, (np.nan, np.nan))[0])
    gen_df["logp"] = gen_df["smiles"].map(lambda x: desc_dict.get(x, (np.nan, np.nan))[1])
    gen_df = gen_df.dropna(subset=["mw", "logp"])

    # Ensure assets dir exists
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Configure Matplotlib Style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    targets = sorted(list(gen_df["target"].unique()))
    correlation_stats = []

    for target in targets:
        print(f"Generating plots for target: {target}...")
        target_ref = ref_df[ref_df["target"] == target]
        target_gen = gen_df[gen_df["target"] == target]
        
        # Build dataset for distribution violin plots
        dist_data_list = []
        if not target_ref.empty:
            dist_data_list.append(target_ref[["model", "mw", "logp", "source"]])
        
        # ─── PLOT FOR EACH EXPERIMENT TYPE (UNCAPPED & CAPPED) ───
        for exp_type in ["Uncapped", "Capped"]:
            exp_gen = target_gen[target_gen["exp_type"] == exp_type]
            
            # MW vs Norm Score - 2x5 Grid
            fig_mw, axes_mw = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
            fig_mw.suptitle(f"Molecular Weight vs. Normalized Docking Score - {target} ({exp_type} Runs)", y=0.98)
            axes_mw_flat = axes_mw.flatten()
            
            # LogP vs Norm Score - 2x5 Grid
            fig_lp, axes_lp = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
            fig_lp.suptitle(f"LogP vs. Normalized Docking Score - {target} ({exp_type} Runs)", y=0.98)
            axes_lp_flat = axes_lp.flatten()
            
            target_ref_valid = target_ref.dropna(subset=["mw", "norm_score"])
            
            # Calculate Reference correlations
            r_mw_ref = np.nan
            r_lp_ref = np.nan
            if len(target_ref_valid) > 2:
                try:
                    r_mw_ref, _ = pearsonr(target_ref_valid["mw"], target_ref_valid["norm_score"])
                    r_lp_ref, _ = pearsonr(target_ref_valid["logp"], target_ref_valid["norm_score"])
                except Exception:
                    pass
            
            # Plot reference on reference axes (Subplot 0)
            if not target_ref_valid.empty:
                axes_mw_flat[0].scatter(target_ref_valid["mw"], target_ref_valid["norm_score"], color="gray", alpha=0.7, s=15)
                label_text = f"Ref r: {r_mw_ref:.2f}" if not pd.isna(r_mw_ref) else ""
                if label_text:
                    axes_mw_flat[0].text(0.05, 0.95, label_text, transform=axes_mw_flat[0].transAxes, verticalalignment="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
                
                axes_lp_flat[0].scatter(target_ref_valid["logp"], target_ref_valid["norm_score"], color="gray", alpha=0.7, s=15)
                label_text = f"Ref r: {r_lp_ref:.2f}" if not pd.isna(r_lp_ref) else ""
                if label_text:
                    axes_lp_flat[0].text(0.05, 0.95, label_text, transform=axes_lp_flat[0].transAxes, verticalalignment="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            
            axes_mw_flat[0].set_title("Reference Set")
            axes_mw_flat[0].set_ylabel("Normalized Docking Score")
            axes_lp_flat[0].set_title("Reference Set")
            axes_lp_flat[0].set_ylabel("Normalized Docking Score")
            
            # Subplots 1 to 9: Models in order
            for idx, model in enumerate(MODEL_ORDER):
                ax_mw = axes_mw_flat[idx + 1]
                ax_lp = axes_lp_flat[idx + 1]
                
                model_gen = exp_gen[exp_gen["model"] == model]
                model_gen_valid = model_gen.dropna(subset=["mw", "norm_score"])
                
                # Pearson correlations
                r_mw_gen, r_lp_gen = np.nan, np.nan
                if len(model_gen_valid) > 2:
                    try:
                        r_mw_gen, _ = pearsonr(model_gen_valid["mw"], model_gen_valid["norm_score"])
                        r_lp_gen, _ = pearsonr(model_gen_valid["logp"], model_gen_valid["norm_score"])
                    except Exception:
                        pass
                
                # Save stats
                correlation_stats.append({
                    "target": target,
                    "exp_type": exp_type,
                    "model": model,
                    "n_gen": len(model_gen),
                    "mean_mw_gen": model_gen["mw"].mean(),
                    "mean_mw_ref": target_ref["mw"].mean() if not target_ref.empty else np.nan,
                    "mean_logp_gen": model_gen["logp"].mean(),
                    "mean_logp_ref": target_ref["logp"].mean() if not target_ref.empty else np.nan,
                    "r_mw_gen": r_mw_gen,
                    "r_mw_ref": r_mw_ref,
                    "r_logp_gen": r_lp_gen,
                    "r_logp_ref": r_lp_ref,
                })
                
                # Plot MW vs Norm Score (Only generated points, in color)
                if not model_gen_valid.empty:
                    ax_mw.scatter(model_gen_valid["mw"], model_gen_valid["norm_score"], color="#1f77b4", alpha=0.6, s=15)
                ax_mw.set_title(f"{model}")
                if idx + 1 >= 5:
                    ax_mw.set_xlabel("Molecular Weight (Da)")
                if (idx + 1) % 5 == 0:
                    ax_mw.set_ylabel("Normalized Docking Score")
                label_text = f"Gen r: {r_mw_gen:.2f}"
                ax_mw.text(0.05, 0.95, label_text, transform=ax_mw.transAxes, verticalalignment="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
                
                # Plot LogP vs Norm Score (Only generated points, in color)
                if not model_gen_valid.empty:
                    ax_lp.scatter(model_gen_valid["logp"], model_gen_valid["norm_score"], color="#2ca02c", alpha=0.6, s=15)
                ax_lp.set_title(f"{model}")
                if idx + 1 >= 5:
                    ax_lp.set_xlabel("LogP")
                if (idx + 1) % 5 == 0:
                    ax_lp.set_ylabel("Normalized Docking Score")
                label_text = f"Gen r: {r_lp_gen:.2f}"
                ax_lp.text(0.05, 0.95, label_text, transform=ax_lp.transAxes, verticalalignment="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            
            # Clean up axes labels
            for k in range(5):
                axes_mw[0, k].set_xlabel("")
                axes_lp[0, k].set_xlabel("")
            for r in range(2):
                for col_idx in range(1, 5):
                    axes_mw[r, col_idx].set_ylabel("")
                    axes_lp[r, col_idx].set_ylabel("")
            
            # Save correlation plots
            fig_mw.tight_layout()
            fig_lp.tight_layout()
            fig_mw.savefig(assets_dir / f"{target}_{exp_type.lower()}_mw_correlation.png")
            fig_lp.savefig(assets_dir / f"{target}_{exp_type.lower()}_logp_correlation.png")
            plt.close(fig_mw)
            plt.close(fig_lp)

        # Collect data for target distributions (capped vs uncapped)
        for model in MODEL_ORDER:
            model_uncap = target_gen[(target_gen["model"] == model) & (target_gen["exp_type"] == "Uncapped")]
            model_cap = target_gen[(target_gen["model"] == model) & (target_gen["exp_type"] == "Capped")]
            if not model_uncap.empty:
                dist_data_list.append(model_uncap[["model", "mw", "logp", "source"]])
            if not model_cap.empty:
                dist_data_list.append(model_cap[["model", "mw", "logp", "source"]])

        # ─── 2. PLOT DISTRIBUTIONS ───
        if dist_data_list:
            comb_dist_df = pd.concat(dist_data_list, ignore_index=True)
            fig_dist, axes_dist = plt.subplots(2, 1, figsize=(14, 10))
            fig_dist.suptitle(f"Property Distributions - {target}", y=0.98)
            
            # Violin for Molecular Weight
            sns.violinplot(ax=axes_dist[0], data=comb_dist_df, x="model", y="mw", hue="source", split=True, inner="quart", gap=0.1, palette={"Reference": ".7", "Generated (Uncapped)": "#1f77b4", "Generated (Capped)": "#aec7e8"})
            axes_dist[0].set_title("Molecular Weight Distribution Comparison")
            axes_dist[0].set_xlabel("")
            axes_dist[0].set_ylabel("MW (Da)")
            axes_dist[0].tick_params(axis="x", rotation=15)
            
            # Violin for LogP
            sns.violinplot(ax=axes_dist[1], data=comb_dist_df, x="model", y="logp", hue="source", split=True, inner="quart", gap=0.1, palette={"Reference": ".7", "Generated (Uncapped)": "#2ca02c", "Generated (Capped)": "#98df8a"})
            axes_dist[1].set_title("LogP Distribution Comparison")
            axes_dist[1].set_xlabel("Model / Group")
            axes_dist[1].set_ylabel("LogP")
            axes_dist[1].tick_params(axis="x", rotation=15)
            
            fig_dist.tight_layout()
            fig_dist.savefig(assets_dir / f"{target}_property_distributions.png")
            plt.close(fig_dist)

    # Compile Stats
    stats_df = pd.DataFrame(correlation_stats)

    # ─── GENERATE MARKDOWN DOCUMENT ───
    md_content = """# Chemical Property & Docking Score Correlation Analysis

This report investigates the correlation between **Normalized Docking Scores** and key physical-chemical descriptors: **Molecular Weight (MW)** and **Octanol-Water Partition Coefficient (LogP)**. 
Additionally, we evaluate how the distribution of these properties differs between **Generated Molecules** (specifically those with a positive docking normalization score $>0.0$) and the corresponding biological target's **Reference Set** (from target ChEMBL bioactivity baseline).

---

## 1. Overview of Key Findings

Historically, generative models optimizing solely for docking score can fall into the trap of "optimizing for size or lipophilicity," because larger and more lipophilic compounds can make non-specific, hydrophobic contacts in a binding pocket, leading to artificially inflated binding energy scores. 

By analyzing the correlation coefficient (Pearson $r$) and looking at the distributions, we can check:
1. **Size Bias**: Do generated molecules have systematically larger Molecular Weights than known active reference compounds?
2. **Lipophilicity Bias**: Do they display abnormally high LogP values?
3. **Descriptor-Docking Correlation**: Is there a strong linear correlation between molecular properties and docking scores in the generated vs. reference sets?

### Reporting Pearson $r$ vs $R^2$ (Coefficient of Determination)
For diagnostic evaluation of descriptor bias, reporting the Pearson correlation coefficient ($r$) is preferred over $R^2$ due to two primary statistical reasons:
1. **Preservation of Direction**: Pearson $r$ ranges from $[-1, 1]$, where the sign indicates the *direction* of the relationship. A positive $r$ directly shows that higher molecular weight or lipophilicity is associated with higher docking scores (positive bias). $R^2$ is bounded $[0, 1]$ and discards directionality, making it impossible to distinguish between size creep (positive correlation) and size reduction (negative correlation).
2. **Linear Association Strength**: Pearson $r$ measures the strength of the linear association, which is the direct metric of interest when diagnosing size/hydrophobic biases in docking scoring functions. $R^2$ describes the proportion of explained variance, which is more applicable to predictive regression modelling than bivariate correlation analysis.

---

## 2. Correlation & Distribution Analysis by Target

"""

    for target in targets:
        t_stats = stats_df[stats_df["target"] == target].copy()
        
        md_content += f"### Target: {target}\n\n"
        
        # Embed distribution figure if it exists
        dist_img = assets_dir / f"{target}_property_distributions.png"
        if dist_img.exists():
            md_content += f"#### Property Distributions (Reference vs Generated)\n"
            md_content += f"![{target} Property Distributions](assets/correlation/{target}_property_distributions.png)\n\n"
        
        # Embed correlation figures in a carousel
        md_content += f"#### Correlation Scatter Plots (MW & LogP vs Normalized Docking Score)\n"
        md_content += "````carousel\n"
        md_content += f"![{target} Uncapped MW vs Docking Score](assets/correlation/{target}_uncapped_mw_correlation.png)\n"
        # slide
        md_content += "<!-- slide -->\n"
        md_content += f"![{target} Capped MW vs Docking Score](assets/correlation/{target}_capped_mw_correlation.png)\n"
        # slide
        md_content += "<!-- slide -->\n"
        md_content += f"![{target} Uncapped LogP vs Docking Score](assets/correlation/{target}_uncapped_logp_correlation.png)\n"
        # slide
        md_content += "<!-- slide -->\n"
        md_content += f"![{target} Capped LogP vs Docking Score](assets/correlation/{target}_capped_logp_correlation.png)\n"
        md_content += "````\n\n"
        
        # Generate statistics table for this target
        md_content += "#### Statistics Summary\n"
        md_content += """| Model | Experiment Type | N (Gen > 0) | Mean MW (Gen) | Mean MW (Ref) | Mean LogP (Gen) | Mean LogP (Ref) | MW Correlation (Gen r) | MW Correlation (Ref r) | LogP Correlation (Gen r) | LogP Correlation (Ref r) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        # Sort t_stats by model order and then uncapped vs capped
        t_stats["model_cat"] = pd.Categorical(t_stats["model"], categories=MODEL_ORDER, ordered=True)
        t_stats = t_stats.sort_values(by=["model_cat", "exp_type"])
        
        for _, row in t_stats.iterrows():
            r_mw_g = f"{row['r_mw_gen']:.2f}" if not pd.isna(row['r_mw_gen']) else "N/A"
            r_mw_r = f"{row['r_mw_ref']:.2f}" if not pd.isna(row['r_mw_ref']) else "N/A"
            r_lp_g = f"{row['r_logp_gen']:.2f}" if not pd.isna(row['r_logp_gen']) else "N/A"
            r_lp_r = f"{row['r_logp_ref']:.2f}" if not pd.isna(row['r_logp_ref']) else "N/A"
            mean_mw_g = f"{row['mean_mw_gen']:.1f}" if not pd.isna(row['mean_mw_gen']) else "N/A"
            mean_mw_r = f"{row['mean_mw_ref']:.1f}" if not pd.isna(row['mean_mw_ref']) else "N/A"
            mean_lp_g = f"{row['mean_logp_gen']:.2f}" if not pd.isna(row['mean_logp_gen']) else "N/A"
            mean_lp_r = f"{row['mean_logp_ref']:.2f}" if not pd.isna(row['mean_logp_ref']) else "N/A"
            
            md_content += f"| {row['model']} | {row['exp_type']} | {row['n_gen']} | {mean_mw_g} | {mean_mw_r} | {mean_lp_g} | {mean_lp_r} | {r_mw_g} | {r_mw_r} | {r_lp_g} | {r_lp_r} |\n"
            
        md_content += "\n---\n\n"

    # Write out the final analysis file
    with open(output_md_path, "w") as f:
        f.write(md_content)
    print(f"Successfully compiled correlation analysis to {output_md_path}")

if __name__ == "__main__":
    main()
