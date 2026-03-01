# Towards Practical and Realistic Evaluation of Chemical Language Models in Drug Discovery

Shu-Hang Lin<sup>1</sup>, Brandon Novy<sup>1</sup>, Konstantin I. Popov<sup>1</sup>

<sup>1</sup>Center for Integrative Chemical Biology and Drug Discovery, Chemical Biology and Medicinal Chemistry, Eshelman School of Pharmacy, University of North Carolina, Chapel Hill, North Carolina 27599, United States

## Abstract

[Abstract: 150-250 words summarizing the problem, approach, key results, and implications. Should concisely state: (1) limitations of current benchmarks, (2) novel docking-based approach, (3) key advantages, (4) main findings/contributions]

## Introduction

Chemical space is vast 
Traditional virtual screening, even with ways to tranverse the increasing large make-on-demand libraries, still only represents a fraction of the estimated drug-like chemical space 
This makes generative models for chemical space exploration a very attractive option 
Because of this, multiple generative models have emerged throughout the years, demonstrating promising capabilities in generating novel chemical structures while adhering to imposed scoring functions (explain). 
Specifically, chemical language models (CLMs), along with concurrent advancements in deep learning, have emerged as powerful tools for molecular generation {Morgan 2025} 
While these approaches offer a more efficient exploration of chemical space, a critical gap has emerged 
A major challenge exists in not having realistic benchmarks for molecular generation {Cieplinski 2023, Coley 2020, MolScore} 
 

## Related Works

To systematically evaluate and compare diverse generative models, the research community has developed several benchmarking frameworks, with GuacaMol and MOSES achieving particularly wide adoption {Brown 2019; Polykovskiy 2020}. GuacaMol's 20 benchmark tasks were subsequently implemented in the popular Therapeutic Data Commons (TDC) suite {Huang 2021}. However, GuacaMol's generative tasks were insufficiently challenging and failed to clearly separate top-performing models {GuacaMol}. 

To address this limitation, Gao et al. introduced MolOpt, adding sample efficiency to the evaluation {Gao 2022}. By constraining each generative task to a budget of 10,000 oracle calls, MolOpt provided a more realistic and challenging metric that better reflects the constraints of real-world drug discovery campaigns. The benchmark was expanded to include 19 GuacaMol tasks (excluding “Aripiprazole Similarity”), three target-based optimization tasks (DRD2, GSK3β, JNK3) from the TDC suite, and the QED scoring function {Gao 2022; QED paper}. Since its introduction, MolOpt has become a widely adopted standard, with numerous subsequent models benchmarking themselves against it, including SAFE-GPT, f-RAG, GenMol, PrexSyn, and ACEGEN {include SAFE-GPT paper; f-RAG paper; GenMol paper; PrexSyn; ACEGEN}. 

However, MolOpt inadvertently rewards the generation of chemically unlikely molecules {Thomas 2022 Re-evaluating}. This is in accordance with findings of Renz et al. in which they showed generative models can act as adversaries that identify where the scoring function extrapolates poorly {Renz 2019}. When applied to similarity- and QSAR-based benchmarks, the models often generate molecules that satisfy the scoring constraints while lacking true drug-like properties {Langevin 2022}. Thomas et al. introduced medicinal chemistry criteria to MolOpt, and showed that the rankings of these models changed substantially, suggesting that the inclusion of orthogonal scoring functions or filters, such as a synthesizability metric or medicinal chemistry filters, is necessary for practical and realistic evaluation of CLMs {Thomas 2022 Re-evaluating}. 

More fundamentally, these benchmarks do not reliably predict the performance of models on more realistic objectives such as predicting binding affinity via structure-based approaches {Thomas 2025 TTT; Thomas 2024 ACEGEN}. To address these limitations, several docking benchmarks have emerged, recognizing that physics-based evaluation offers significant advantages over purely statistical approaches. Notable benchmarks include DOCKSTRING {García-Ortegón 2022} and smina-docking-benchmark {Cieplinski 2023}. These initiatives reflect growing recognition that a docking oracle provides a more realistic proxy for drug discovery campaigns, especially for data-scarce targets where insufficient ligand data exists to train robust predictive models {Thomas 2021 GPCR}. Moreover, structure-based scoring functions generate more diverse, bioactive-like chemistry compared to ligand-based oracles and enable the exploration of novel chemical space unconstrained by the bias toward previously established chemotypes inherent in ligand-based approaches {Thomas 2021 GPCR}. 

However, existing docking benchmarks have shortcomings that limit their utility. First, models optimizing on docking scores can lead to the generation of large and/or highly lipophilic molecules {García-Ortegón 2022}. Such molecules are undesirable from a medicinal chemistry perspective due to poor ADMET properties and increased risk of promiscuous binding {Thomas 2024 MolScore}. This means that docking benchmarks using score alone as the objective will prioritize impractical molecules for the purpose of advancing a drug discovery campaign {Thomas 2024 MolScore}. Second, very few of the existing docking benchmarks implement comprehensive ligand preparation protocols. These should include protonation at biologically relevant pH, stereoisomer, and tautomer enumeration {ten Brink 2009; Bender 2021}. MolScore unifies diverse scoring functions and addresses these concerns by including ligand preparation protocols {Thomas 2024 MolScore}. Third, to our knowledge, none of the existing docking benchmarks consider the predicted binding pose in their scoring function. Instead, they report only the final scalar docking scores, neglecting the geometric accuracy, or inaccuracy, of the predicted binding poses, which is an essential component of structure-based drug design {Jones 1997; Trott 2010}. In a structure-based drug discovery campaign, it would be highly risky to propose synthesizing and testing molecules that have good docking scores with an unrealistic or physically impossible binding pose. 

These molecular generation benchmarks serve as essential infrastructure for progress in the field, enabling fair comparison across models, driving algorithmic innovation, and identifying weaknesses in current approaches that must be addressed before generative AI can achieve meaningful impact in drug discovery. The need for more realistic and rigorous benchmarks has been repeatedly emphasized in the literature {Coley 2020; others?}. Physics-based evaluation through molecular docking represents a logical middle ground between fast, simplistic similarity-based benchmarks and time-consuming and costly experimental testing. More importantly, docking is routinely employed in early-stage drug discovery for hit identification, hit expansion, and hit-to-lead optimization {Cieplinski 2023; include docking in early discovery paper}. Moreover, with the advent of geometric deep learning for molecular graphs and the rapid development of structure-based generative models employing roto-translationally equivariant neural networks, molecular docking has become increasingly important both as a scoring function to guide generation and as a natural evaluation metric for structure-aware approaches {Cieplinski 2023; include geometric DL papers}.  In this work, we address the identified limitations by introducing a physics-based benchmark that combines fragment constraints derived from real medicinal chemistry campaigns with rigorous structural validation, providing an evaluation of both predicted binding affinity and geometric accuracy of binding poses. 

## Fragment-Constrained Molecular Generation

There has been increasing emphasis on fragment-constrained generation—the ability to generate molecules containing specific structural fragments. The SAFE (Sequential Attachment-based Fragment Embedding) representation was developed specifically to enable this capability, with SAFE-GPT being the first model trained on this representation {SAFE paper}. Subsequently, f-RAG and GenMol introduced architectural improvements while maintaining the SAFE string format to preserve fragment-constrained generation capabilities. Traditional recurrent neural network approaches, such as REINVENT and Link-INVENT, achieve fragment constraints by conditioning generation on input scaffold fragments. 

The emphasis on fragment-constrained generation is well-motivated by both theoretical foundations and practical drug discovery considerations. Fragment-Based Drug Discovery (FBDD) is based on the "Complexity Model," which postulates that "there is a higher probability of a match between a ligand and its receptor if there are fewer interactions to get right."27 Because fragments are simple (typically <300 Da), they exhibit high "Ligand Efficiency" (binding energy per atom), making them ideal starting points for optimization.27,28 Over the past 25 years, FBDD has emerged as a highly successful approach, delivering approved drugs such as Vemurafenib and Venetoclax.28 

Medicinal chemistry campaigns frequently require molecules containing specific scaffolds for multiple reasons: (1) maintaining structural features essential for target binding based on structure-activity relationship (SAR) studies; (2) preserving pharmacokinetic properties associated with particular chemical moieties; (3) navigating intellectual property constraints; and (4) building upon validated chemical starting points from high-throughput screening or fragment-based drug discovery.27,29,30 

Furthermore, there exists a significant disconnect between the academic focus on de novo design and industrial practice, which is overwhelmingly dominated by Lead Optimization (LO). Lead optimization is the iterative process of refining a "hit" molecule to improve its potency, selectivity, and ADME properties, and "finding a molecule with the ideal ratio of these characteristics... is the ultimate objective."19,20 The preference for LO over de novo design is driven by risk mitigation and resource constraints: while computational models can generate billions of structures, "the bottleneck in lead optimization is the synthetic chemistry," and de novo models often produce structures that are synthetically inaccessible or require complex, unverified routes.21,22 Industrial campaigns rarely start from zero but begin with high-throughput screening (HTS) hits or fragment hits. 

## Our Contribution

In this work, we introduce the Fragment-Constrained Generative Model Benchmark (FCGMB), a novel benchmarking framework designed to evaluate chemical language models in scenarios that more closely reflect the challenges of practical drug discovery. Our approach addresses the limitations of existing benchmarks by:

1. **Integrating structure-based evaluation** through molecular docking against experimentally validated crystal structures, providing physics-based validation that is significantly harder to exploit than QSAR models;$^{35,40,41}$

2. **Requiring fragment-constrained generation** based on maximum common substructures from real medicinal chemistry series, aligning with industrial practice of lead optimization and FBDD;$^{27,29}$

3. **Evaluating both binding pose accuracy and activity prediction** simultaneously using the industry-standard 2.0 Å RMSD criterion established by seminal docking validation studies;$^{35,38,40}$

4. **Using diverse, literature-derived chemical series** to create realistic optimization scenarios that avoid the "known answer" problem of similarity-based approaches; and

5. **Providing a scalable, fully open-source framework** that can grow with the PDB and ChEMBL databases, using accessible docking software (AutoDock-GPU and AutoDock Vina) to ensure reproducibility.

By combining fragment constraints derived from real medicinal chemistry campaigns with structure-based validation through molecular docking, our benchmark provides a more holistic and realistic assessment of CLM capabilities. This approach not only evaluates whether models can generate molecules similar to known actives (the focus of current benchmarks) but whether they can generate novel molecules that maintain key structural features while achieving favorable predicted binding modes—a task that mirrors the actual workflow of structure-based drug design. 

## Methods
 
### Target and Structure Identification
 
Benchmark construction began with a systematic retrieval of all human single-protein targets from the ChEMBL36 database. For each identified target, we queried the Protein Data Bank (PDB) using UniProt accession identifiers to obtain associated structures and their bound ligands. Structures determined by X-ray crystallography, cryo-electron microscopy (cryo-EM), and nuclear magnetic resonance (NMR) spectroscopy were all included without discrimination by experimental methods. This cross-referencing between ChEMBL and PDB ensured that each benchmark entry possessed both experimental bioactivity data and structural information suitable for structure-based evaluation. 
 
### Ligand Drug-Likeness Filtering
 
Bound ligands from holo structures were subjected to stringent drug-likeness filtering to ensure benchmark compounds exhibited properties representative of drug-like chemical space. Ligands were retained only if they satisfied the following criteria: molecular weight between 200 and 800 Da, at least one ring system, presence of nitrogen or oxygen heteroatoms, calculated LogP (cLogP) between −2 and 7, and no more than 15 rotatable bonds. Only holo structures (containing bound ligands) were retained for downstream analysis. This filtering step employed RDKit molecular descriptors to eliminate non-drug-like fragments and experimental artifacts. 
 
### Document Association and Compound Filtering
 
For each reference ligand passing drug-likeness criteria, we queried the ChEMBL database to identify associated assay documents containing binding activity measurements (assay_type = 'B') with valid pChEMBL values. To ensure statistical robustness in benchmarking tasks, we retained only assays containing at least 20 compounds with quantified bioactivity. Documents were matched to PDB structures through molecular similarity analysis using Morgan circular fingerprints (radius 2, 2048 bits) with a Tanimoto similarity threshold of 0.99, thereby identifying exact matches of the reference ligand within a document's compound series. Assay data were organized at the individual assay level (assay_chembl_id) to guarantee bioactivity measurements originated from consistent experimental conditions. Document source types (literature vs. other) were annotated based on ChEMBL source identifiers. 
 
### Benchmark Configuration Generation
 
To ensure reliable pose validation, only structures with a reported resolution of 3.0 Å or better were retained for benchmark configuration generation. For selected compounds within each assay, we computed the Maximum Common Substructure (MCS) relative to the reference ligand using RDKit's FMCS (Find Maximum Common Substructure) module. To ensure that the identified MCS represented a true common scaffold across the chemical series, we computed the MCS across up to 100 compounds per assay with a 100% presence threshold—requiring the substructure to be present in all selected molecules. The FMCS algorithm enforced strict matching criteria: atom types and bond orders were required to match exactly (AtomCompare.CompareElements, BondCompare.CompareOrder), valences were enforced (matchValences=True), and ring systems were preserved as complete rings with ring-to-ring matches only (ringMatchesRingOnly=True, completeRingsOnly=True). A 10-second timeout was applied per assay to prevent computational bottlenecks on particularly complex series. 
 
Clean SMILES representations of the MCS were extracted by matching the SMARTS pattern against the crystal ligand template and applying kekulization to generate chemically valid fragment SMILES without aromatic markers. To ensure that retained MCS fragments were structurally meaningful, additional quality filters were applied: the MCS SMILES must be syntactically valid, must not contain generic bond notation (tildes, '~'), and must have a molecular weight exceeding 90 Da (aka “High Quality MCS” in Table 1). These thresholds eliminated trivially small or invalid substructures that would provide insufficient constraint during molecule generation. Each benchmark configuration was then defined by a unique combination of a ChEMBL document identifier, a ChEMBL assay identifier, an RCSB PDB identifier, and the MCS represented as a SMILES string, forming the structural constraint for molecule generation. 
 
### Receptor Preparation
 
Structures were downloaded in mmCIF format from the RCSB PDB. ProDy was employed to parse mmCIF files and locate the primary instance of the bound ligand, handling alternate conformations by preferring blank altloc designations or selecting the most common altloc when necessary. Using the ligand's 3D coordinates as a reference point, we identified and extracted all protein chains containing atoms within 5.0 Å of any ligand atom, ensuring that relevant binding site residues were retained while excluding distant protein chains unnecessary for rigid-receptor docking. The resulting receptor complex was exported as a PDB file after removing all organic ligands, waters, and solvent molecules. The reference ligand was exported as a separate PDB file used in Autogrid4. 
 
The receptor underwent further refinement using Reduce2 (mmtbx.reduce2 from the CCTBX suite) to add hydrogen atoms and optimize the hydrogen-bond network. Grid maps for molecular docking were subsequently generated using Meeko's mk_prepare_receptor utility (with --allow_bad_res to handle non-standard residues) in combination with AutoGrid4. The docking grid was centered on the reference ligand position with 5.0 Å padding in all directions to define the search space. 
 
### Ligand Preparation
 
Ligand data retrieved from ChEMBL included ChEMBL compound identifiers, canonical SMILES strings, and bioactivity measurements expressed as pChEMBL values (negative log of molar IC50, EC50, Ki, or Kd values). Bioactivity values were preprocessed at the assay level: duplicate SMILES entries within the same assay were consolidated using the median pChEMBL value. The ligands underwent a comprehensive multi-step preparation pipeline to ensure chemically valid and biologically relevant conformations: 
 
Standardization: RDKit was employed in conjunction with the MolStandardize module to strip salts (retaining only the largest fragment), neutralize charges where chemically sensible (Uncharger), and canonicalize SMILES representations, ensuring consistency across molecular representations. 
 
Tautomer and Ionization State Generation: MolScrub was utilized to enumerate appropriate tautomeric forms and ionization states within a physiologically relevant pH range of 6.4 to 8.4, approximating conditions in typical biochemical assays. A maximum of 16 distinct protonation/tautomeric states were retained per ligand to balance thoroughness with computational tractability. 
 
Conformer Generation: Three-dimensional conformers were generated using RDKit's ETKDGv3 (Experimental Torsion Knowledge Distance Geometry version 3) algorithm with small ring torsions enabled (useSmallRingTorsions=True), allowing exploration of realistic low-energy conformations. When initial embedding failed, random coordinate generation was employed as a fallback strategy. A single low-energy conformation was retained per state. 
 
Docking Input Generation: Final PDBQT files were prepared using Meeko's mk_prepare_ligand utility. Ligand preparation was parallelized across available CPU cores using Python's multiprocessing module with the spawn context to ensure thread-safe execution. A total limit of 32 prepared states/conformers per unique compound was enforced to maintain computational efficiency while capturing conformational and ionization state diversity. 
 
### Score Normalization and Model-Visible Compounds

To simulate a realistic drug discovery campaign, we split each benchmark's bioactivity data into two sets. The **model-visible set** (lower 25% of the pChEMBL activity range) represents weakly active compounds analogous to early-stage screening hits that would be available to a medicinal chemistry team. This set is provided to generative models as initialization data via `get_initial_compounds()`. The remaining **validation set** (upper 75%) contains the more potent compounds that the generative model is tasked with rediscovering or surpassing. This split simulates the realistic scenario in which a drug discovery campaign begins with weak hits and the objective is to optimize toward higher potency, rather than having access to the most active compounds from the outset.

The activity threshold is computed as: threshold = min(pChEMBL) + 0.25 × (max(pChEMBL) − min(pChEMBL)), applied per-assay to ensure the split reflects the specific activity distribution of each chemical series.

Docking scores are normalized to a [0, 1] scale using empirically determined bounds:

$$\text{normalized\_score} = \frac{\text{low\_score} - \text{raw\_score}}{\text{low\_score} - \text{high\_score}}$$

where **low_score** is the centroid (mean) of all model-visible compounds' docking scores across 5 variance runs, representing the baseline expected performance, and **high_score** is the best (most negative) mean docking score observed across all compounds in 5 variance runs, representing the achievable ceiling. Scores can exceed 1.0 if a generated molecule achieves a better docking score than any known compound from the ChEMBL series.

| Benchmark | low_score | high_score | Score Range |
|-----------|-----------|------------|-------------|
| VEGFR2    | -10.05    | -12.36     | 2.31        |
| PCK1      | -10.76    | -13.35     | 2.59        |
| ITK       | -8.83     | -11.09     | 2.26        |
| TTK       | -9.83     | -12.74     | 2.91        |
| AKT1      | -8.79     | -13.87     | 5.08        |
| CHK1      | -7.63     | -12.52     | 4.89        |

### Docking Campaign
 
A large-scale docking campaign was executed using either AutoDock-GPU to leverage GPU acceleration for high-throughput performance or AutoDock Vina with the AD4 scoring function when GPU resources were unavailable. Hardware detection was performed automatically: when AutoDock-GPU was available and GPU resources were detected, the GPU backend was prioritized; otherwise, the workflow defaulted to Vina. For benchmarks evaluated with AutoDock Vina, we specified an exhaustiveness parameter of 32 and retained up to 10 poses per docking run. AutoDock-GPU similarly generated 10 poses per ligand (--nrun 10). All docking was performed against the AutoGrid4 maps generated during receptor preparation. Docking was executed across all prepared benchmark configurations, with each ligand docked into the binding site defined by its corresponding reference structure. Prior to docking, a two-dimensional substructure filter was applied: only compounds containing the MCS fragment constraint (verified using RDKit substructure matching) were submitted for docking, ensuring that evaluated compounds represented genuine analogs of the reference. 
 
### Structural Validation
 
The accuracy of docked poses was evaluated by calculating the Root Mean Square Deviation (RMSD) of the maximum common substructure (MCS) between each docked ligand and the reference ligand from the experimentally determined structure. The MCS was matched in both the docked pose and the reference structure using RDKit's substructure matching algorithms, with robust fallback to loosened query parameters (generic bond matching, aromaticity adjustment) to handle tautomeric differences between docked and experimental states. RMSD was computed over atomic coordinates of the matched fragment atoms. A threshold of 2.0 Å was established as the criterion for a "passed" docking pose, consistent with the gold standard established by seminal docking validation studies. 
 
The 2.0 Å threshold has historical origins dating to the development of the first robust automated docking algorithms. In the development of the Genetic Optimization for Ligand Docking (GOLD) program, Jones et al. defined a successful docking prediction as one where the top-ranked pose was within 2.0 Å of the reference, demonstrating that genetic algorithms could reliably reproduce experimental binding modes within this tolerance. Similarly, Morris et al., in their description of the Lamarckian Genetic Algorithm for AutoDock 3.0, utilized the 2.0 Å RMSD criterion to evaluate the convergence of their search method. The continued relevance of this standard was reinforced by Trott and Olson in the validation of AutoDock Vina, which remains one of the most widely used docking tools today. 
 
This distance typically corresponds to the resolution limits of many protein-ligand crystal structures and the tolerance for forming hydrogen bonds (2.5–3.5 Å). An RMSD deviation of less than 2.0 Å implies that the ligand has maintained the "native" binding mode, preserving critical polar and hydrophobic interactions. In rigorous benchmarks like the CASF-2016 study, top-performing docking tools (e.g., Glide, GOLD, Surflex) typically achieve success rates (RMSD < 2.0 Å) of 50–90% depending on target complexity. As established in the docking literature, RMSD < 2.0 Å corresponds to "good" or successful docking solutions, 2.0–3.0 Å indicates "acceptable" orientation with possible atomic interaction shifts, and RMSD > 3.0 Å represents "bad" solutions that are effectively random or trapped in local minima. 
 
For each ligand, the docking result selection followed an RMSD-first, score-second protocol: the best-scoring pose among those passing the RMSD threshold was selected. If no poses passed the 2.0 Å threshold, the best-scoring pose regardless of RMSD was reported with an invalid pose flag, enabling downstream analysis of both pose quality and scoring performance. 
 
### Performance Metrics
 
Docking performance for each benchmark was assessed by correlating docking scores with experimental bioactivity values (pChEMBL). We calculated several statistical metrics to comprehensively evaluate both structural and scoring accuracy: 
 
- Percent Passed: The percentage of compounds achieving a docked pose within the 2.0 Å RMSD threshold, reflecting the ability of the docking protocol to reproduce crystallographic binding modes. 
 
- Coefficient of Determination (R²): Measures the proportion of variance in experimental bioactivity explained by docking scores, indicating the utility of docking scores for activity prediction. 
 
- Pearson Correlation Coefficient (r): Assesses the linear relationship between docking scores and bioactivity values, quantifying the strength and direction of correlation. 
 
- Spearman's Rank Correlation Coefficient (ρ): Evaluates monotonic relationships between docking scores and bioactivity, providing a non-parametric alternative robust to outliers and non-linear relationships. 
 
These metrics collectively provide a comprehensive assessment of benchmark quality, enabling evaluation of docking programs across both geometric accuracy (pose prediction) and energetic accuracy (affinity ranking). 
 
## Results
 
### Benchmark Dataset Construction
 
#### Target and Structure Identification
 
Systematic mining of the ChEMBL36 database identified 5,820 human single-protein targets with curated bioactivity data. Cross-referencing these targets with the RCSB Protein Data Bank (accessed February 4, 2026) via UniProt accession identifiers yielded 63,250 unique structures associated with 3,884 targets (66.7% of all human single-protein targets). Among these structures, 40,771 (64.5%) were holo structures containing bound ligands, representing 56,519 unique PDB-ligand combinations. 
 
#### Drug-Likeness Filtering
 
Application of drug-likeness criteria substantially refined the dataset to focus on pharmacologically relevant chemical matter. Of the initial 23,080 unique ligand entries, 18,441 passed the drug-likeness filter. The drug-likeness filter thus eliminated experimental fragments, cofactors, and non-drug-like molecules. 
 
#### Bioactivity Data Integration
 
Cross-referencing filtered PDB-ligand pairs with ChEMBL bioactivity data identified 633 targets with 6,023 structures associated with 4,841 assays from 3,950 documents. For each reference ligand, we queried ChEMBL for associated assay documents containing binding activity measurements (assay_type = 'B') with valid pChEMBL values. Only assays with at least 20 compounds with quantified bioactivity were retained to ensure statistical robustness. Documents were matched to PDB structures through molecular similarity analysis using Morgan circular fingerprints (radius 2, 2048 bits) with a Tanimoto similarity threshold of 0.99. [Insert exact number of total unique compounds across all retained assays. Insert distribution statistics: median/mean compounds per assay, range.] 
 
#### Maximum Common Substructure Analysis
 
Maximum common substructure (MCS) computation was successful for 417 targets across 4,281 structures and 3,784 assays. The FMCS algorithm with strict matching criteria (exact atom type and bond order matching, valence enforcement, complete ring preservation) identified structurally meaningful scaffolds shared across each chemical series. After applying quality filters requiring valid SMILES, molecular weight exceeding 90 Da, and resolution ≤ 3.0 Å, 327 targets with 2,882 structures and 2,267 assays remained. [Insert distribution of MCS molecular weights and atom counts across the retained configurations. Insert number of assays filtered at each sub-step.] 
 
**Table 1.** Summary statistics of data mining, filtering, and dataset construction.

| Data Processing Stage | Targets | PDB Structures | PDB-Ligand Pairs | Ligands | Assays / Documents |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I. Initial Data Retrieval** | | | | | |
| ChEMBL36 Human Single-Protein Targets | 5,820 | N/A | N/A | N/A | N/A |
| Cross-referencing Targets with PDB | 3,884 | 63,250 | N/A | N/A | N/A |
| Holo Structure Identification | 2,674 | 40,771 | 56,519 | 23,080 | N/A |
| **II. Drug-Likeness Filtering** | | | | | |
| Filtering for Drug-Like Properties | 1,842 | 27,630 | 30,287 | 18,441 | N/A |
| **III. Bioactivity Integration** | | | | | |
| Document Association & Filtering | 633 | 6,023 | 6,051 | 4,548 | 4,841 / 3,950 |
| **IV. MCS Analysis** | | | | | |
| Calculable MCS | 417 | 4,281 | 4,296 | 3,191 | 3,784 / 3,059 |
| **V. Additional Filtering** | | | | | |
| High Quality MCS & 3.0Å Resolution | 327 | 2,882 | 2,890 | 2,240 | 2,267 / 1,892 |
 
By mapping each assay series to all available compatible PDB structures, we generated 7,792 unique docking configurations for validation. [Insert: Of these 7,792 configurations, X were successfully docked, Y failed at various stages (retrieval: __, PDB 404: __, ligand missing: __, grid prep: __, reference match: __, docking: __, analysis: __). The overall success rate was X/7792 = __%. Insert total wall-clock time for the full campaign and average seconds per conformer.]

### Benchmark Selection via Variance Analysis

From the initial 7,792 docking configurations, we computed Spearman rank correlation coefficients between docking scores and experimental pChEMBL values. The top 30 configurations ranked by Spearman correlation were selected for variance testing to assess docking reproducibility. Each of these 30 systems was docked 5 independent times using identical receptor grids and ligand preparation protocols, producing 150 total docking campaigns.

[Insert: Total compounds docked across all 150 campaigns. Total conformers docked. Average and stdev of Spearman across the 5 runs for each system.]

From the variance analysis, six benchmark tasks were manually selected based on three criteria: (1) **consistency**—low standard deviation of Spearman correlation across runs (σ < [insert threshold]); (2) **correlation strength**—high mean Spearman correlation (|ρ| > 0.74 for all selected benchmarks); and (3) **docking score dynamic range**—sufficient spread between the best and worst docking scores to enable meaningful discrimination (e.g., a range of −9 to −10 kcal/mol was considered insufficient, whereas ranges of 4–5 kcal/mol were preferred). This ensures that the benchmark scoring function can meaningfully rank compounds across a broad activity spectrum.
 
 
By mapping each assay series to all available compatible PDB structures, we generated 7,792 unique docking configurations for validation. This redundancy ensures that chemical series are tested against multiple structural conformations of the target when available (and vice versa: multiple chemical series sharing a high-quality MCS with the reference ligand are docked into the same structure). 
### Docking Performance Analysis

[Insert: Of the 7,792 docking configurations attempted, X completed successfully (XX.X%). Y configurations failed at the following stages: retrieval (n=__), PDB 404 (n=__), ligand not found (n=__), grid preparation (n=__), reference ligand match failure (n=__), docking failure (n=__), and analysis failure (n=__). Successful configurations encompassed Z unique targets across W unique PDB structures.]

[Insert: Total number of unique compounds docked across all successful configurations. Total number of conformers docked. Average docking time per conformer (seconds). Total wall-clock time for the full docking campaign.]

[Present distribution of performance metrics across all benchmarks: histograms or violin plots showing distribution of Percent Passed, R², Pearson correlation, Spearman correlation. Discuss what these distributions reveal about benchmark difficulty and quality]

**[Figure 2]** [PLACEHOLDER: Histograms or violin plots of metric distributions across all 7,792 configurations: (A) Percent Passed RMSD, (B) Pearson r, (C) Spearman ρ, (D) R².]

**[Figure 3]** [PLACEHOLDER: Ranked bar plot of mean Spearman correlation for the top 30 systems with error bars from 5 variance runs. Highlights the 6 selected benchmarks.]

### Case Studies

[Present 2-4 exemplary benchmark tasks in detail: show the crystal structure, the MCS constraint, the chemical series from the literature, docking results for known actives, correlation plots. Discuss what makes these good benchmark tasks] 

[Figures showing specific examples with molecular structures, binding site visualizations, and performance plots] 

**Table 2.** Exemplary benchmark tasks from the FCGMB dataset.

| Target Name (ID) | PDB ID | Resolution (Å) | Fragment SMILES | N Compounds | Mean Spearman ρ | % Passed RMSD | Score Range (low → high) | Assay Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VEGFR2 (CHEMBL279) | 3VHE | 1.55 | C1=CC2=C(C=NC=N2)N1 | 24 | −0.790 | 100% | −10.05 → −12.36 | [Ki/IC50] |
| PCK1 (CHEMBL2911) | 1NHX | 2.10 | [insert] | 21 | −0.854 | 100% | −10.76 → −13.35 | [Ki/IC50] |
| ITK (CHEMBL2959) | 3QGW | 2.10 | CC1=CNC(=O)C=C1 | 22 | −0.826 | 100% | −8.83 → −11.09 | [Ki/IC50] |
| TTK (CHEMBL3983) | 3WZJ | 2.75 | [insert] | [n] | −0.741 | [%] | −9.83 → −12.74 | [Ki/IC50] |
| AKT1 (CHEMBL4282) | 4EJN | 2.19 | C1=CN=CC(C2=NC3=CC=CN=C3N2)=C1 | [n] | −0.747 | 100% | −8.79 → −13.87 | [Ki/IC50] |
| CHK1 (CHEMBL4630) | 2R0U | 1.90 | O=C1NC=CC2=C1C1=C(C=CC=C1)C=C2 | 33 | −0.837 | 100% | −7.63 → −12.52 | [Ki/IC50] |

### Figures

**[Figure 1]** [PLACEHOLDER: Workflow diagram of the complete pipeline from ChEMBL/PDB data mining → drug-likeness filtering → document association → MCS computation → receptor preparation → grid generation → ligand preparation → docking → structural validation → scoring. Show branch for pre-built grids vs. on-the-fly preparation.]

**[Figure 4]** [PLACEHOLDER: Panel of scatter plots (2×3 grid) showing pActivity (pChEMBL) vs. mean docking score from 5 variance runs for each of the 6 selected benchmarks. Each panel should show error bars (score std across runs), lower-25% model-visible compounds in orange, upper-75% validation compounds in blue, and crystal ligand highlighted as a diamond. Include Spearman ρ annotation in each panel.]

**[Figure 5]** [PLACEHOLDER: Panel of 2D MCS + 3D MCS-in-pocket visualization for each of the 6 benchmarks. Left: 2D depiction of the MCS fragment highlighted within a representative compound. Right: 3D rendering of the MCS atoms (colored/highlighted) overlaid with the crystal ligand pose in the protein binding pocket. Show protein surface or ribbon with key interactions.]

**[Figure 6]** [PLACEHOLDER: Panel showing the top-scoring molecule generated by each generative model for each benchmark task. Include 2D structure, docking score, normalized score, and whether the pose passed the 2.0 Å RMSD threshold. Organized as a grid: rows = models, columns = benchmarks.]

**[Figure 7]** [PLACEHOLDER: Crystal ligand re-docking validation. For each of the 6 benchmarks, show overlay of re-docked pose (colored) vs. crystal pose (gray), with RMSD value annotated. Demonstrates that the docking protocol can reproduce known binding modes.]

**[Figure 8]** [PLACEHOLDER: Score distribution comparison showing the normalized score distributions for each generative model across all 6 benchmarks. Violin or box plots with individual data points overlaid.]
 
 
 
 
 

## Comparison with Existing Benchmarks

[Discuss how the structure-based evaluation with pose validation provides additional information: e.g., molecules can have good docking scores but physically implausible poses. This is only caught by RMSD validation against the crystal reference.]

**Table 3.** Comparison of molecular generation benchmark characteristics.

| Feature | GuacaMol | MOSES | MolOpt | DOCKSTRING | smina-dock | MolScore | FCGMB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Evaluation Type | Similarity | Distribution | Multi-objective | Docking | Docking | Multi-objective | Docking + Pose |
| Fragment Constraint | No | No | No | No | No | Optional | Yes (MCS) |
| Pose Validation | No | No | No | No | No | No | Yes (2.0 Å RMSD) |
| Ligand Preparation | N/A | N/A | N/A | Basic | Basic | Yes | Yes (Scrub, tautomers, ionization) |
| Open Source | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| N Tasks | 20 | 1 | 23 | [n] | [n] | Configurable | 6 (expandable) |
| Budget Constraint | No | No | 10,000 | No | No | Configurable | 5,000 (configurable) |
| Physics-Based | No | No | Partial | Yes | Yes | Partial | Yes |
| Score Normalization | N/A | N/A | N/A | Raw score | Raw score | Configurable | Empirical bounds |

## Evaluation of State-of-the-Art Models

### Models Evaluated

We evaluated [N] state-of-the-art chemical language models spanning diverse architectural paradigms:

**Autoregressive / RNN-Based Models:**
- **REINVENT** {Blaschke 2020}: A recurrent neural network (RNN) using transfer learning and reinforcement learning for goal-directed molecular generation. Uses SMILES as its molecular representation.
- **Link-INVENT** {Guo 2023}: Fragment-linking extension of REINVENT that generates molecules by connecting user-provided fragments via a conditional RNN.

**Transformer-Based Models:**
- **SAFE-GPT** {Noutahi 2024}: A GPT-based model trained on the SAFE (Sequential Attachment-based Fragment Embedding) representation, naturally enabling fragment-constrained generation through its sequential fragment attachment paradigm.
- **f-RAG** {f-RAG paper}: A retrieval-augmented generation framework using SAFE strings that retrieves relevant molecular fragments during generation to improve chemical validity and property optimization.
- **GenMol** {GenMol paper}: [Brief description of architecture and key innovation.]

**Other Architectures:**
- **PrexSyn** {PrexSyn}: [Brief description.]
- **ACEGEN** {Thomas 2024 ACEGEN}: [Brief description.]

[Insert additional models as applicable.]

### Benchmarking Protocol

Each model was given a budget of 5,000 oracle calls per benchmark task. Models received as input: (1) the fragment SMILES constraint that all generated molecules must contain, (2) the model-visible compound set (lower 25% of pChEMBL activity range) as initialization data. Models were tasked with generating novel molecules that (a) contain the specified fragment, (b) achieve favorable docking scores, and (c) produce binding poses within 2.0 Å RMSD of the crystal reference.

[Insert: Describe any model-specific configuration details, e.g., learning rates, number of epochs, sampling temperatures.]

### Benchmarking Results

**Table 4.** [PLACEHOLDER: Generative model benchmarking results. Columns: Model, AUC-Top10 (per benchmark), Mean AUC-Top10, Top-1 Score, % Valid (containing fragment), % Passed RMSD, Oracle Calls Used, Wall-Clock Time.]

**Table 5.** [PLACEHOLDER: Molecular quality metrics for each generative model. Columns: Model, Validity (%), Uniqueness (%), Novelty (%), Internal Diversity (Tanimoto), Fragment Retention Rate (%), Median MW, Median cLogP, Median QED.]

**Table 6.** [PLACEHOLDER: Raw docking scores from each generative model per benchmark. Columns: Model, Benchmark, Best Score, Median Score, Mean Score, % Poses < 2.0 Å RMSD.]

## Discussion

### Advantages of Structure-Based Benchmarking

Our structure-based benchmarking framework offers several key advantages over existing approaches. First, it provides more realistic evaluation that mirrors structure-based drug design workflows used in industrial settings, where structure-activity relationships are explored through crystallographic and computational methods.23,26,33 Second, it avoids the "known answer" problem inherent in similarity-based benchmarks, where the goal is to regenerate existing molecules rather than discover novel chemical matter. 

Third, our approach evaluates both activity and binding mode simultaneously, providing a more stringent test of model capabilities.35,41 Fourth, it provides interpretable feedback through 3D structural analysis, allowing researchers to understand why generated molecules succeed or fail at a mechanistic level. Fifth, it can identify failure modes not captured by 2D similarity metrics, such as molecules that appear similar by fingerprint comparison but adopt completely different binding modes.17,18 

Critically, structure-based methods enable scaffold hopping—the identification of novel scaffolds that fit the protein pocket but look topologically distinct from known binders.32,33 This capability is essential for bypassing patent constraints and discovering First-in-Class therapeutics. In contrast, ligand-based models that rely on the similarity principle are inherently biased toward scaffolds present in their training data.32 

Perhaps most importantly, by using physics-based docking scores derived from Van der Waals forces and electrostatics rather than statistical correlations, our framework provides a robust check against the "QSAR hacking" problem that plagues similarity-based and QSAR-based benchmarks.6,11,47 The 2.0 Å RMSD criterion creates a "physics-based adversary" that is significantly harder to exploit than trained predictors. 

### Integration with Fragment-Constrained Generation

Our benchmark is particularly suited for evaluating fragment-constrained CLMs such as SAFE-GPT, f-RAG, GenMol, and REINVENT. The maximum common substructure provides a natural structural constraint that these models can use as input, enabling evaluation of a critical capability for practical drug discovery.27,29,30 

Experimental protocols for benchmarking would provide each model with the MCS fragment and task it with generating molecules that: (1) contain the MCS, (2) achieve favorable docking scores, and (3) produce binding poses within 2.0 Å RMSD of the crystal reference. This mirrors the three primary fragment evolution strategies used in FBDD: fragment growing (adding chemical groups to reach adjacent sub-pockets), fragment merging (chemically fusing overlapping fragments), and fragment linking (connecting fragments bound to distinct sites).29,30 Models could be compared on their ability to efficiently explore chemical space under these constraints, measured by metrics such as the number of valid proposals needed to discover molecules passing all criteria. 

### Limitations and Future Directions

Our framework has several limitations that should be acknowledged. First, docking scores are approximations of binding affinity, not experimental measurements. While more physically grounded than QSAR predictions, they cannot fully capture all factors influencing binding, such as entropy effects and induced fit. Methods like free energy perturbation (FEP) would provide higher-quality estimates but are computationally prohibitive for the scale required in benchmarking thousands of molecules.47 

Second, our approach uses a rigid receptor assumption, which may miss binding modes that require protein conformational changes. Third, the framework is limited to targets with available crystal structures, excluding important therapeutic targets like GPCRs and ion channels that are difficult to crystallize. Fourth, we focus exclusively on human proteins, potentially missing important benchmarks from other organisms relevant to infectious disease or veterinary medicine. 

Fifth, we strip all waters during receptor preparation, potentially missing essential waters necessary for activity. However, this also makes the benchmark more robust and reproducible across different docking protocols. Sixth, we use a standard pH range of 6.4-8.4 for protonation state generation, but not all parts of the human body maintain pH 7.4. While it would be more accurate to identify protein localization and adjust pH accordingly, this level of detail would significantly complicate the benchmark without proportional benefit. 

Seventh, these benchmarks take longer to run than traditional ones based on similarity calculations, as docking requires 3D structure generation and energy minimization. However, we mitigate this through GPU acceleration (AutoDock-GPU) and provide both fast (AutoDock-GPU) and standard (AutoDock Vina) protocols. Eighth, the 5 Å padding for the docking box might not accommodate very large generated molecules, though this constraint also serves to focus generation on drug-like chemical space. 

Finally, there is potential for data leakage since we use publicly available data from ChEMBL and PDB. Models trained on datasets overlapping with our benchmark structures might have an unfair advantage. We do not currently include 3D generative models in our evaluation, which could be addressed in future work. 

Future directions include: (1) incorporation of protein flexibility through ensemble docking or molecular dynamics; (2) multi-objective optimization including ADME properties and synthetic accessibility alongside binding affinity; (3) expansion to other organisms and target classes; (4) integration of experimental validation campaigns to assess whether computationally high-scoring molecules show activity in vitro; and (5) crowd-sourcing of improved docking protocols and scoring functions as they become available.47,58 

### Scalability and Community Contributions

[The benchmark is designed to scale with growing structural and bioactivity databases. As new protein-ligand co-crystal structures are deposited in the PDB and new bioactivity data accumulates in ChEMBL, the data mining pipeline (available in `pipeline/`) can be re-run to generate additional benchmark configurations. The current release covers 6 manually curated benchmarks, but the 7,792 validated docking configurations provide a reservoir of candidate benchmarks for future expansion. Community members can contribute new benchmarks by providing YAML configuration files specifying target_id, pdb_id, doc_id, assay_id, ligand_resname, and fragment_smiles.]

[Additionally, while we provide AutoDock-GPU and AutoDock Vina as docking backends, the modular architecture (with an abstract `DockingOracle` base class) readily accommodates alternative docking engines such as Glide, GOLD, or rDock. Cross-platform validation using commercial docking software (e.g., Schrödinger Glide, OpenEye FRED) would further strengthen confidence in the benchmark's conclusions.]

### Implications for Model Development

Given that CLMs have demonstrated remarkable ability to optimize oracle functions when properly incentivized,4 this benchmark reveals that the key bottleneck in generative model performance is not model architecture but rather the quality of the oracle itself. Our structure-based approach provides a higher-quality oracle than similarity metrics or QSAR models, potentially driving development of models that are more directly applicable to real drug discovery. 

This may require models to evolve beyond simple sequence-to-sequence generation to incorporate geometric awareness and 3D structural understanding.26,34 We emphasize that our framework is fully open-source—utilizing freely available data from ChEMBL and RCSB PDB, open-source docking software (AutoDock-GPU and AutoDock Vina), open-source cheminformatics libraries (RDKit), and Python for reproducibility. We provide benchmarking protocols for both AutoDock-GPU and AutoDock Vina to ensure accessibility across different computational environments, and we welcome crowd-sourced contributions of improved docking protocols as the field advances. 

## Conclusion

Current benchmarks for evaluating chemical language models do not adequately assess their capabilities in realistic drug discovery scenarios. The reliance on similarity-based metrics and QSAR models creates evaluation frameworks that are vulnerable to exploitation and disconnected from industrial practice, where lead optimization and fragment-based approaches dominate over de novo design. 

We have introduced a novel benchmarking framework that combines fragment constraints derived from real medicinal chemistry campaigns with structure-based validation through molecular docking. By requiring generated molecules to maintain key structural features (the MCS) while achieving physically plausible binding modes (RMSD < 2.0 Å from crystal reference), our benchmark provides a more stringent and realistic test of CLM capabilities. 

The key advantages of our approach include: physics-based evaluation that is harder to exploit than statistical predictors, alignment with industrial workflows emphasizing lead optimization and FBDD, simultaneous assessment of both binding pose and predicted affinity, interpretable 3D structural feedback, and ability to identify failure modes invisible to 2D similarity metrics. By providing this benchmark as a fully open-source resource built on publicly available data and software, we aim to drive development of CLMs that can more effectively contribute to the discovery of new therapeutics. 

Ultimately, our vision is that rigorous, realistic benchmarking will accelerate the translation of generative AI from algorithmic novelty to practical impact in drug discovery, helping to bring life-saving medicines to patients faster and more efficiently. 

## References

1. Polykovskiy, D. et al. (2020). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models. Frontiers in Pharmacology. 
2. Brown, N. et al. (2019). GuacaMol: Benchmarking Models for de Novo Molecular Design. J. Chem. Inf. Model.. 
3. Fréchet ChemNet Distance: A metric for generative models for molecules. PMC. 
4. Stereochemistry-aware string-based molecular generation. PNAS Nexus, 4(11). 
5. Impact of applicability domains to generative artificial intelligence. ResearchGate. 
6. Impact of Applicability Domains to Generative Artificial Intelligence. ACS Omega. 
7. GuacaMol Benchmark for Molecular Design. Emergent Mind. 
8. Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models. ACS JCIM. 
9. Bajorath, J. et al. (2022). Activity cliffs limitations of 2D similarity Tanimoto. J. Med. Chem.. 
11. Renz, P. et al. (2019). On failure modes in molecule generation and optimization. Drug Discovery Today: Technologies. 
13. Advanced Artificial Intelligence Technologies Transforming Contemporary Pharmaceutical Research. MDPI. 
14. Turk et al. Identifying Potential Missteps of Machine Learning in Molecular Chemistry. ChemRxiv. 
15. Evolving Concept of Activity Cliffs. ACS Omega. 
16. Maggiora, G.M. (2006). On outliers and activity cliffs—why QSAR often disappoints. J. Chem. Inf. Model.. 
17. Stumpfe, D., Hu, Y., Dimova, D., & Bajorath, J. (2014). Recent progress in understanding activity cliffs. J. Med. Chem.. 
18. Stumpfe, D., & Bajorath, J. (2012). Exploring activity cliffs in medicinal chemistry. J. Med. Chem.. 
19. De novo drug design through artificial intelligence: an introduction. Front. Hematol.. 
20. Lead Optimization: Integrating Experimental, Computational, and AI/ML Approaches. IJPS Journal. 
21. Efficient Drug Lead Discovery and Optimization. PMC. 
22. Computational Methods in Drug Discovery and Development. ChemRxiv. 
23. Schneider, G. (2018). Automating drug discovery. Nature Reviews Drug Discovery. 
25. Chemists: AI Is Here; Unite To Get the Benefits. Pure Manchester. 
26. Structure-Based Drug Design with a Deep Hierarchical Generative Model. ACS JCIM. 
27. What makes a good fragment in fragment-based drug discovery? Taylor & Francis. 
28. Fragment-based drug discovery: A graphical review. PMC. 
29. The Virtual Elaboration of Fragment Ideas. Cresset Group. 
30. Fragment-based Lead Preparation in Drug Discovery. Life Chemicals. 
32. Identification of CYP1A2 ligands by structure-based and ligand-based virtual screening. ACS JCIM. 
33. Structure-based methods provide detailed information about specific protein-ligand interactions. PMC. 
34. Employing Molecular Conformations for Ligand-Based Virtual Screening. MDPI. 
35. Jones, G. et al. (1997). Development and validation of the program GOLD. J. Mol. Biol.. 
37. GOLD validation: Top-scoring answers were correct for 71 out of 100 complexes. DIVA Portal. 
38. Morris, G.M. et al. (1998). Automated docking using a Lamarckian genetic algorithm. J. Comput. Chem.. 
40. Trott, O. & Olson, A.J. (2010). AutoDock Vina: improving the speed and accuracy of docking. J. Comput. Chem.. 
41. Validation of Molecular Docking: The RMSD must be lower than 2.0 Å. Front. Nutr.. 
43. Software for molecular docking: a review. PMC. 
45. Is It Reliable to Take the Molecular Docking Top Scoring Position? Molecules. 
47. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function. PMC. 
58. AL and uncertainty-aware exploration offer a principled alternative. PMC. 
## Data Availability

All benchmark data, evaluation scripts, and protocols are publicly available through a GitHub repository (https://github.com/Popov-Lab-UNC/fcgmb) and will be archived on Zenodo for long-term accessibility. The benchmark package includes: (1) curated YAML configuration files defining each benchmark task; (2) pre-built AutoGrid4 maps for all six benchmark receptor structures; (3) curated bioactivity data CSVs with ChEMBL compound identifiers and pChEMBL values; (4) the `FCGMBOracle` Python class providing a standardized scoring interface; (5) reference ligand coordinates with corrected bond orders for RMSD validation; and (6) complete data mining and receptor preparation pipelines for generating new benchmarks. The package is installable via `pip install fcgmb` (or `pip install -e .` from source) and requires only AutoDock-GPU or AutoDock Vina as an external dependency for standard usage. All other software dependencies (RDKit, Meeko, MolStandardize) are installed automatically.

## Supplementary Information

### S1. Detailed Ligand Preparation Protocol

The multi-step ligand preparation pipeline ensures chemically valid and biologically relevant docking inputs:

1. **Standardization**: RDKit's `MolStandardize` module strips salts (retaining the largest fragment via `LargestFragmentChooser`), neutralizes charges (`Uncharger`), and canonicalizes SMILES representations.

2. **Tautomer and Ionization State Enumeration**: MolScrub enumerates tautomeric forms and ionization states within a physiologically relevant pH range of 6.4–8.4. A maximum of 16 distinct protonation/tautomeric states are retained per compound.

3. **Stereoisomer Handling**: Input stereochemistry is preserved when specified. When `generate_isomers=True` (default), MolScrub enumerates stereoisomers; when `False`, only the input stereoisomer is retained. This is relevant for models that operate on isomeric SMILES.

4. **3D Conformer Generation**: ETKDGv3 with `useSmallRingTorsions=True` generates one 3D conformer per state. Random coordinate fallback is employed if distance geometry embedding fails (e.g., for strained ring systems).

5. **PDBQT Conversion**: Meeko's `MoleculePreparation` and `PDBQTWriterLegacy` convert RDKit Mol objects to PDBQT format, assigning Gasteiger charges and AutoDock atom types. A maximum of 32 prepared states/conformers per unique compound is enforced.

6. **Parallelization**: Ligand preparation is parallelized across available CPU cores using Python's `multiprocessing` module with the `spawn` context to ensure thread-safe execution.

### S2. Bioactivity Data Curation

Bioactivity data for each benchmark is bundled as pre-curated CSV files within the package (`fcgmb/bioactivity_data/<benchmark>.csv`). Each CSV contains three columns: `molecule_chembl_id`, `canonical_smiles`, and `pchembl_value` (−log₁₀ of IC₅₀, Ki, or Kd). Data was curated at the assay level (single `assay_chembl_id`) to guarantee measurements originate from consistent experimental conditions. Duplicate SMILES entries within the same assay were consolidated using the median pChEMBL value.

The oracle supports three data sources in order of priority: (1) bundled CSV (fastest, no network required), (2) local scratch cache (`.fcgmb/data/<name>_chembl.csv`), and (3) live ChEMBL API fetch (used only when no cached data exists).

### S3. Software Versions and Dependencies

[Insert exact versions used for benchmarking: AutoDock-GPU version, AutoDock Vina version, AutoGrid4 version, Reduce2/CCTBX version, Meeko version, RDKit version, MolScrub version, Python version. Insert hardware specifications: GPU model(s), number of CPUs, RAM.]

### S4. YAML Configuration File Format

Each benchmark is defined by a YAML configuration file containing all parameters needed to reproduce the docking campaign:

```yaml
benchmark_name: AKT1                    # Human-readable benchmark name
pdb_id: 4EJN                            # RCSB PDB structure identifier
target_id: CHEMBL4282                   # ChEMBL target identifier
doc_id: CHEMBL2176970                   # ChEMBL document identifier
assay_id: CHEMBL2186223                 # ChEMBL assay identifier
ligand_resname: 0R4                     # PDB residue name of the crystal ligand
fragment_smiles: C1=CN=CC(...)=C1       # MCS fragment (clean SMILES)
fragment_smiles_with_dummies: [*]C1=... # MCS with dummy attachment points
require_fragment_match: true            # Enforce 2D substructure filter
require_pose_rmsd: true                 # Enforce 3D RMSD validation
low_score: -9.741                       # Score normalization lower bound
high_score: -14.342                     # Score normalization upper bound
rmsd_threshold: 2.0                     # RMSD threshold for pose validation (Å)
```

### S5. Complete Variance Analysis Results

[Insert: Full table of variance analysis for all 30 systems tested: system key, n_compounds, n_runs, mean_pearson (±σ), mean_r² (±σ), mean_spearman (±σ). Highlight the 6 selected benchmarks.]

## Acknowledgments

[Acknowledgments of funding sources, computational resources, and individuals who contributed] 

## Competing Interests

The authors declare no competing interests. 
 
https://pubs.acs.org/doi/10.1021/acs.jcim.4c00519?ref=PDF 
 
 
GuacaMol: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839 
Sample Efficiency Matters (aka PMO or mol-opt): https://arxiv.org/pdf/2206.12411 
Benchmarking 3D: https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.5c01020?ref=article_openPDF 
Test-Time Training Scaling Laws: https://pubs.acs.org/doi/10.1021/acs.jcim.5c02316 (super useful paper that goes over what I try to talk about) 
ExCAPE-DB: https://link.springer.com/article/10.1186/s13321-017-0203-5 
PrexSyn: https://arxiv.org/abs/2512.00384 
 
Surrogate models is more important, generative models going OOD is bad: https://www.nature.com/articles/s41524-025-01924-8 
 
## Miscellaneous Notes

- **3D-focused benchmarks**: More recently, 3D-focused benchmarks such as GenMolBench and GenBench3D have emerged.
- **Software Accessibility**: We include AutoDock Vina for accessibility. To note, the original benchmarking was done in AutoDock-GPU. Most generative models require GPU for generation, so we will assume users looking to benchmark have access to GPUs. But in the case that the use of AutoDock-GPU is unavailable, fallback to AutoDock Vina is included (slowdown of ~33x).
- **Cross-platform validation**: Show that the docking models are good using Schrodinger and OpenEye too.
- **Software versions**: AutoDock-GPU, AutoDock Vina, AutoGrid4, Reduce2, etc.
 
## Comparative Discussion

### Similarity-based vs. Target-based Benchmarks

Similarity-based benchmarks evaluate models on their ability to regenerate molecules that are already known to be active. In actual medicinal chemistry campaigns, however, the goal is to discover novel chemical matter with desired properties, not to rediscover existing solutions. This fundamental mismatch means that high performance on similarity-based benchmarks may not translate to success in prospective molecular design {Coley 2020; Thomas 2025 TTT}. The target-based benchmarks in GuacaMol (GSK3β, DRD2, JNK3) partially address this concern by using machine learning classifiers trained on bioactivity data from the ExCAPE-DB database {Sun 2017; Brown 2019}. While these provide more activity-relevant evaluation than pure similarity metrics, they introduce new concerns about prediction reliability for out-of-distribution molecules. When a generative model produces a molecule structurally distinct from the classifier's training set, the predicted activity may be profoundly unreliable, potentially rewarding models for generating molecules that appear active according to flawed predictions but would fail in experimental validation {Renz 2019; Langevin 2022}.

### Advantages of the FCGMB Benchmark

The FCGMB benchmark represents a substantial expansion over existing structure-based benchmarking resources. Unlike static datasets such as DUD-E or PDBbind that focus on receptor–ligand binding prediction, FCGMB provides fragment-constrained generation tasks with paired bioactivity labels, enabling simultaneous evaluation of structural fidelity (via RMSD-based pose validation) and property prediction (via correlation with experimental pChEMBL values). The inclusion of 7,792 distinct benchmark configurations—each with a defined chemical scaffold constraint derived from experimental structural data—offers unprecedented scale and diversity for systematic assessment of fragment-based and scaffold-hopping generative approaches.

Furthermore, the assay-level organization ensures that bioactivity comparisons are made within consistent experimental contexts, avoiding the confounding effects of mixed assay conditions that plague many existing benchmarks. The requirement that the reference ligand itself appear in the assay compound series guarantees that each benchmark possesses a known "ground truth" binding mode, facilitating rigorous pose prediction evaluation alongside affinity prediction.
 