MOCKDOCK: A Practical Benchmark for Chemical Language Models in Structure-Based Drug Design

Shu-Hang Lin1, Brandon Novy1, Konstantin I. Popov1

1 Center for Integrative Chemical Biology and Drug Discovery, Chemical Biology and Medicinal Chemistry, Eshelman School of Pharmacy, University of North Carolina, Chapel Hill, North Carolina 27599, United States

## Abstract

Chemical language models (CLMs) are increasingly evaluated with benchmarks that are either too similarity-driven or too dependent on surrogate property predictors. Both settings can reward models for recovering known chemistry or for exploiting scoring functions outside their domain of validity, rather than for proposing realistic structure-based ideas under medicinal chemistry constraints. We introduce MOCKDOCK, a fragment-constrained, docking-based benchmark designed to better reflect lead optimization and fragment-based drug discovery (FBDD). Each task is built from a matched PDB structure and ChEMBL assay series, and models must generate molecules that retain a benchmark-specific maximum common substructure (MCS) while improving structure-based score. Generated molecules are subjected to the same preparation stack as benchmark compounds, including tautomer and protonation-state enumeration, 3D conformer generation, MMFF94s/UFF minimization, docking, and pose validation against the crystal ligand using a 2.0 A fragment RMSD threshold. We further report post hoc medicinal chemistry metrics and filtered optimization metrics using PAINS, BMS, and physicochemical filters. From a large-scale mining pipeline, we identify six robust benchmark tasks whose docking scores correlate strongly with experimental pChEMBL and remain stable across five repeated variance runs. Initial experiments on AceGen variants and PrexSyn show that PPO-style RL methods are the strongest optimizers under a fixed oracle budget, whereas PrexSyn offers excellent validity and diversity but struggles to satisfy the fragment and pose-constrained objective. MOCKDOCK provides a practical, open, and sample-efficient test bed for evaluating CLMs in realistic structure-based design settings.

## 1. Introduction

The central promise of molecular generative modeling is sample-efficient exploration of enormous chemical design spaces. In practice, however, medicinal chemistry is not an unconstrained search problem. Campaigns are budget-limited, highly local, and typically proceed from an experimentally grounded series rather than from unrestricted de novo generation. This is especially true in lead optimization and FBDD, where chemists often preserve a core fragment or scaffold while exploring substitutions, growth vectors, or linking strategies.

Existing benchmarks only partially capture this setting. Similarity-based suites such as GuacaMol and MOSES are useful for measuring basic generation quality, but they do not directly test whether a model can improve a structure-based objective under realistic constraints. QSAR-driven benchmarks such as MolOpt move closer to practical optimization and emphasize sample efficiency, yet they still depend on learned surrogate models that can often be exploited by strong generators. This creates the familiar reward-hacking problem: models discover regions where the scorer extrapolates poorly rather than molecules that are genuinely plausible medicinal chemistry proposals.

Docking-based evaluation is a natural next step because it replaces purely statistical oracles with a physics-motivated one. However, docking-only benchmarks remain incomplete if they optimize a single scalar score without checking whether the resulting pose is geometrically credible. They also remain disconnected from day-to-day medicinal chemistry if they allow unconstrained molecular growth that ignores scaffold continuity, fragment retention, and series context.

MOCKDOCK is designed around those practical gaps. The benchmark combines four ingredients that are usually treated separately: (1) matched protein-ligand crystal structures and assay series, (2) an explicit fragment constraint derived from a medicinal chemistry series MCS, (3) docking with fragment-level pose validation, and (4) fixed-budget evaluation that rewards sample-efficient search. In this sense, the benchmark sits at the intersection of modern molecular optimization benchmarks and realistic structure-based lead generation. It also naturally connects to FBDD: preserving the MCS anchors generation around a validated binding motif, while the benchmark objective asks whether a model can elaborate that motif into stronger, still-plausible binders.

## 2. Benchmark Construction

### 2.1. Mining target-structure-assay systems

Benchmark construction starts by querying ChEMBL36 for human single-protein targets and cross-referencing those targets to holo PDB structures through UniProt identifiers. Bound ligands are filtered to remove obvious crystallographic artifacts and non-drug-like matter. For each retained structure, we identify ChEMBL binding assays with valid pChEMBL measurements and sufficient dynamic range to support meaningful ranking. We then match assay series back to the crystallographic reference ligand using a strict fingerprint-similarity requirement, ensuring that each benchmark task is tied to a coherent medicinal chemistry series rather than to loosely related chemistry.

To define the fragment constraint, we compute the MCS shared between the crystallographic ligand and the matched assay series. The MCS is required to be chemically meaningful, fully preserved across the selected series, and free of generic bond notation. This produces a benchmark-specific fragment that acts as a 2D hard constraint during docking and, for prompt-based generators, can also be supplied as a scaffold-like conditioning signal.

At the end of this mining and filtering pipeline, the search space contracts from thousands of targets and tens of thousands of structures to a smaller set of high-quality structure-assay pairs suitable for repeated docking and downstream selection.

**Table 1. Summary statistics of benchmark construction.**

| Stage | Targets | PDB structures | PDB-ligand pairs | Ligands | Documents / assays |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChEMBL36 human single-protein targets | 5,820 | N/A | N/A | N/A | N/A |
| Cross-referenced with PDB | 3,891 | 63,644 | N/A | N/A | N/A |
| Holo structures identified | 2,682 | 41,046 | 56,984 | 23,194 | N/A |
| Drug-like ligand filtering | 1,847 | 27,798 | 30,470 | 18,527 | N/A |
| Document association and assay filtering | 632 | 5,961 | 5,989 | 4,504 | 3,894 / 4,775 |
| Calculable MCS | 401 | 3,747 | 3,761 | 2,769 | 2,771 / 3,363 |
| High-quality MCS and <= 3.0 A resolution | 311 | 2,450 | 2,458 | 1,884 | 1,671 / 1,965 |

### 2.2. Receptor and ligand preparation

Receptors are prepared from mmCIF structures by extracting the protein chains surrounding the bound reference ligand, removing solvent and co-crystallized small molecules, adding hydrogens, and generating AutoDock-compatible grid maps centered on the crystal ligand. In the packaged release, the six final benchmarks are distributed with pre-built grids to make the benchmark inexpensive to use.

Ligands from the benchmark series and model-generated ligands are processed with the same preparation pipeline. SMILES are standardized, then MolScrub enumerates protonation and tautomeric states over pH 6.4-8.4, retaining at most 16 states per input molecule. For each state, we generate a 3D conformer with ETKDGv3, retry with random coordinates if distance geometry fails, and then perform force-field minimization before conversion to PDBQT. When RDKit has full MMFF parameters, we use MMFF94s minimization; otherwise we fall back to UFF. A maximum of 32 prepared ligand states/conformers is retained per unique input molecule. This same stack is applied both during benchmark construction and when evaluating molecules emitted by a generator.

### 2.3. Docking, pose validation, and score normalization

Docking is performed with AutoDock-GPU by default, with AutoDock Vina available as an accessibility-oriented fallback. We generate 10 poses per molecule. Before docking, molecules must pass a hard 2D fragment screen: if a generated molecule does not contain the benchmark fragment, it is charged against the oracle budget but receives a failing score.

Docked poses are evaluated against the crystal ligand using a fragment-centered RMSD calculation. Rather than comparing entire ligands, we align and score the atoms corresponding to the benchmark fragment/MCS, which is the chemically relevant conserved substructure. A pose is considered valid if the fragment RMSD is below 2.0 A. Scoring follows an RMSD-first policy: for each molecule, we record the best docking score among poses that satisfy the RMSD threshold; if no pose passes, the molecule is marked as a pose failure and receives a reward score of 0.0.

Docking scores are normalized with benchmark-specific calibration points. `high_score` is the best (minimum) mean docking score observed across the variance runs for that task, and `low_score` is the worst (maximum) mean docking score observed across those same original benchmark compounds. This produces an uncapped `norm_score` in which higher is better and values outside `[0, 1]` indicate performance outside the original benchmark docking range. The RL-facing `reward_score` is `norm_score` clipped to `[0, 1]`, so reaching 1.0 means matching or exceeding the best original benchmark compound while leaving diversity and novelty to distinguish methods after that point.

The clipping is intentional for reinforcement learning. Several molecular RL algorithms assume non-negative, bounded rewards, and unbounded docking-derived values can change the effective learning problem across targets: negative sentinel scores can dominate policy updates, while very high scores on easy targets can encourage over-optimization of docking energy instead of continued exploration. A bounded `reward_score` makes the optimization objective comparable across benchmarks and gives every method the same success threshold. We still retain the uncapped `norm_score` for post-hoc analysis, so improvements beyond the original benchmark range are visible without giving additional training reward.

### 2.4. Model-visible split, variance analysis, and final task selection

For each benchmark, we split the assay series by the empirical 25th percentile of observed pChEMBL values. Molecules at or below this threshold form the model-visible set. These are the least active compounds in the series and serve as the generator's starting context. The remaining molecules are treated as held-out analogs used to evaluate whether the docking oracle recovers experimental rank structure on stronger chemistry.

This split is also important conceptually: MOCKDOCK is not asking models to memorize the best known analogs. It is asking whether a model can improve from weak or middling chemistry under a fixed experimental budget, which mirrors the sample-efficiency focus emphasized in modern molecular optimization benchmarks.

After large-scale docking over all candidate systems, we keep only systems with more than 75% fragment-RMSD pass rate. The surviving systems are ranked by the Spearman correlation between docking score and experimental pChEMBL. The notebook `benchmark/notebooks/best_pl_systems.ipynb` and its companion CSV are used to inspect these top systems and assemble the shortlist. We then rerun the top 30 systems with `scripts/variance/run_variance.py`, performing five repeated docking runs per system using the same receptor grids and preparation protocol. The final six benchmarks are selected manually from this set using three criteria: high mean Spearman correlation, low variance across the five reruns, and a sufficiently broad docking-score range to discriminate between generators.

## 3. Experimental Setup

### 3.1. Models evaluated

The current benchmark codebase implements three model families relevant to this draft: AceGen, Lib-INVENT, and PrexSyn. Placeholder descriptions for SAFE-GPT, f-RAG, GenMol, and a separate standalone REINVENT baseline were removed because those models are not represented in the present experiment stack.

**AceGen.** AceGen is used here as a family of SMILES-based RL optimizers with a GRU policy and PromptSMILES-style fragment conditioning. We evaluate six optimization variants: A2C, AHC, PPO, PPO with replay/distillation (PPOD), REINFORCE, and a REINVENT-style variant implemented inside the AceGen stack. For each benchmark, the fragment with dummy attachment points is converted into a prompt scaffold, and the model is optimized directly against the MOCKDOCK oracle under a fixed budget.

**Lib-INVENT.** Lib-INVENT is integrated through REINVENT4 staged learning. In this benchmark it operates as a scaffold-decoration model, using the benchmark fragment with at least two attachment points as the conditioning scaffold. The current default configuration disables score-based early stopping (`termination = "null"`) and disables sequence-level deduplication (`unique_sequences = false`) so that oracle accounting more faithfully reflects every generated proposal. At the time of writing, some Lib-INVENT and other model jobs are still queued, so the quantitative comparison below reflects only the completed runs currently available in the analysis outputs.

**PrexSyn.** PrexSyn is used as a learned-projector plus evolutionary optimization method. A neural loader and molecular projector define a latent/fingerprint search space, and optimization proceeds through a genetic algorithm rather than autoregressive sequence RL. To make the comparison fair, the initialization procedure seeds PrexSyn from the model-visible compounds rather than from purely random fingerprints, repeated molecules are still charged against the oracle budget, and the optimization loop runs until budget exhaustion rather than stopping early at an internally satisfactory score.

### 3.2. Benchmark protocol

All generators are evaluated with a budget of 1,000 oracle calls per benchmark. For the completed model experiments summarized here, each model-target pair is run for five independent runs/seeds, and final metrics are reported as means across those runs. All model runs are executed on NVIDIA L40S GPUs. AceGen performs a small warmup on up to 25 model-visible compounds before RL begins, then uses the remaining budget for optimization. PrexSyn likewise initializes around the model-visible set and uses a 1% mutation rate to diversify its starting population. Lib-INVENT uses batch size 64 and `max_steps = ceil(budget / 64)` so that the effective generation schedule matches the same fixed oracle budget.

Crucially, generated molecules are not evaluated with a lighter-weight proxy than the benchmark compounds. Each generated SMILES is subjected to the same 2D fragment check, state enumeration, 3D embedding, MMFF94s/UFF minimization, docking, and fragment-RMSD pose validation described in Section 2.

### 3.3. Evaluation metrics

We evaluate each run with a post hoc evaluator that reports both intrinsic generation metrics and oracle-facing optimization metrics.

Intrinsic and library-quality metrics include validity, uniqueness, internal diversity, scaffold diversity, QED, synthetic accessibility, fragment incorporation, novelty against the model-visible set, and nearest-neighbor similarity to the initial compounds.

Optimization metrics include top-1, top-10, and top-100 normalized scores, the AUC of the running top-10 curve (`AUC-Top10`), the fraction of molecules yielding a valid pose under the RMSD criterion, and `oracle_efficiency_80`, defined as the number of oracle calls required to reach 80% of the final top-10 score.

We also apply post hoc medicinal chemistry filters. A molecule is counted as med-chem passing only if it avoids PAINS and BMS structural alerts and satisfies simple physicochemical bounds (molecular weight 100-700, logP -3 to 6.5, and <= 10 rotatable bonds). We additionally report Lipinski pass rates separately. This distinction matters: a model can optimize the docking oracle while still producing chemistry that is unattractive for follow-up, and the filtered metrics quantify that gap directly.

## 4. Results

### 4.1. From large-scale mining to six benchmark tasks

The mining pipeline yielded 6,976 unique docking configurations, of which 6,840 completed successfully. Filtering for systems with strong fragment-RMSD pass rates and meaningful docking-bioactivity correlation produced a shortlist of 30 systems for repeated reruns. Five variance runs per shortlisted system were then used to estimate score stability and select the final six tasks.

The resulting tasks span six target-structure pairs and show strong negative Spearman correlation between docking score and pChEMBL, indicating that better docking scores generally correspond to stronger measured compounds within each series.

**Table 2. Final MOCKDOCK benchmark tasks.**

| Target | PDB ID | Compounds | Spearman correlation | Passed RMSD (%) | Score range (kcal/mol) |
| --- | --- | ---: | ---: | ---: | --- |
| VEGFR2 | 3VHE | 24 | -0.87 +/- 0.00 | 100.0 +/- 0.0 | -10.04 to -12.41 |
| DPP4 | 2HHA | 36 | -0.86 +/- 0.02 | 100.0 +/- 0.0 | -9.30 to -11.29 |
| ITK | 3QGW | 22 | -0.83 +/- 0.03 | 99.1 +/- 1.8 | -8.00 to -10.09 |
| CHK1 | 2R0U | 44 | -0.83 +/- 0.01 | 76.8 +/- 2.7 | -6.99 to -11.79 |
| PEPCK | 2GMV | 21 | -0.80 +/- 0.02 | 100.0 +/- 0.0 | -9.72 to -11.79 |
| TTK | 3WZJ | 25 | -0.78 +/- 0.01 | 91.2 +/- 4.7 | -8.72 to -11.81 |

Two points are worth emphasizing. First, these are not arbitrary docking targets; they are selected because docking retains experimental rank information on a matched medicinal chemistry series. Second, the final tasks are intentionally not all equally easy. CHK1 and TTK are somewhat less permissive under the pose filter than DPP4, PEPCK, and VEGFR2, which gives the benchmark meaningful variation in difficulty.

### 4.2. Generative model performance under a fixed oracle budget

The current aggregate experiment outputs cover the completed AceGen variant runs and PrexSyn runs across the six benchmarks, each averaged over five runs. Additional model families are still running in the queue. Within the completed runs, a clear pattern emerges: PPO-style AceGen methods are the strongest optimizers, while PrexSyn is strong on syntactic generation quality but weak on the benchmark objective itself.

**Table 3. Macro-averaged model performance across six benchmarks (current analyzed runs).**

| Model | Validity | Fragment incorporation | Avg Top-10 | AUC-Top10 | Valid pose rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| AceGen-A2C | 0.513 | 0.433 | 1.140 | 0.897 | 0.833 |
| AceGen-AHC | 0.473 | 0.437 | 1.059 | 0.798 | 0.867 |
| AceGen-PPO | 0.554 | 0.434 | 1.173 | 0.976 | 1.000 |
| AceGen-PPOD | 0.557 | 0.437 | 1.271 | 0.992 | 0.933 |
| AceGen-REINFORCE | 0.456 | 0.430 | 1.060 | 0.779 | 0.833 |
| AceGen-REINVENT | 0.456 | 0.429 | 1.084 | 0.800 | 0.867 |
| PrexSyn | 1.000 | 0.062 | 0.128 | 0.157 | 0.667 |

AceGen-PPOD is the strongest overall optimizer in the current set, with the best macro-average `Avg Top-10` and `AUC-Top10`, closely followed by AceGen-PPO. This is consistent with the intuition that under a strict 1,000-call budget, methods with stable on-policy optimization and stronger reuse of learning signal should perform best. Across most tasks, the AceGen family also maintains nontrivial fragment incorporation while still improving docking score, indicating that the fragment-constrained objective is learnable rather than merely punitive.

PrexSyn shows the opposite trade-off. It achieves perfect validity and the highest internal diversity, but only a small fraction of its molecules satisfy the required fragment constraint, which collapses its effective optimization performance. In other words, PrexSyn generates chemically valid and diverse molecules, but many of those molecules are not valid proposals for the benchmarked medicinal chemistry problem because they do not preserve the task-defining fragment.

At the target level, VEGFR2 appears especially challenging for the current generators. Several methods produce near-zero or negative normalized `Avg Top-10` values on this benchmark, suggesting that VEGFR2 functions as a useful hard case rather than an outlier to be removed. By contrast, ITK and PEPCK are more consistently improvable and therefore serve as informative mid-difficulty tasks for differentiating strong from weak optimizers.

### 4.3. Why pose-aware and med-chem-aware evaluation matters

A benchmark that reports only the best scalar docking score misses two common failure modes. The first is geometric failure: a generator may produce molecules that score well but do not reproduce a credible fragment placement in the binding site. The second is medicinal chemistry failure: a generator may optimize the oracle while drifting into unattractive property space or alert-rich chemistry.

MOCKDOCK addresses the first failure mode with fragment-centered RMSD validation and the second with post hoc med-chem filtering. These filters do not alter the oracle during optimization; instead, they provide a downstream lens on whether a method is finding chemistry that is both high-scoring and plausibly triageable. This separation is useful because it distinguishes "can the model optimize the task as defined?" from "would a medicinal chemist want to inspect the resulting molecules?" In a benchmark paper, both questions matter.

## 5. Discussion

MOCKDOCK is designed to close the gap between generic molecular generation benchmarks and realistic structure-based design workflows. Its main contribution is not simply the use of docking, but the combination of docking with series-derived fragment constraints, fragment-centered pose validation, and fixed-budget evaluation. Together these choices make the benchmark harder to game and more aligned with real lead-optimization and FBDD practice.

The benchmark also makes a broader methodological point. Better generative models alone are not enough if the evaluation oracle is too easy to exploit. By anchoring tasks in crystal structures and experimentally measured series, and by requiring both plausible geometry and scaffold continuity, MOCKDOCK moves benchmarking closer to the decisions medicinal chemists actually care about.

The early model results are instructive. They suggest that strong budget-aware RL can meaningfully optimize the benchmark, but also that high validity and high diversity alone are not sufficient if the model cannot stay anchored to the required fragment and produce geometrically credible poses. This is exactly the kind of distinction that similarity-based or unconstrained docking benchmarks tend to blur.

## 6. Limitations

MOCKDOCK still inherits the limitations of docking-based evaluation. Docking scores remain approximate and do not account for receptor flexibility, water networks, entropic effects, or downstream developability. The receptors are effectively rigid, and the final six tasks are selected from public structural and assay data, which means some degree of training-data overlap is possible for broadly trained models.

The current benchmark release also emphasizes 2D/sequence-driven generators rather than 3D generative models. That is a reasonable starting point for an open benchmark, but it leaves room for future comparisons against methods with explicit geometric reasoning. Finally, not all model families have finished running at the time of writing, so the present quantitative section should be read as an interim comparison over completed experiments rather than a final exhaustive benchmark leaderboard.

## 7. Conclusion

MOCKDOCK provides a practical benchmark for evaluating CLMs in a setting closer to real structure-based medicinal chemistry. By requiring fragment retention, realistic ligand preparation, docking, fragment-centered pose validation, and fixed-budget sample efficiency, it offers a more stringent alternative to similarity-driven and surrogate-driven benchmarks. The six released tasks provide a stable and interpretable starting point, and the initial model results already show that the benchmark can separate optimization strength from mere syntactic generation quality. We expect this framework to be useful both for benchmarking current CLMs and for motivating future models that better integrate scaffold awareness, sample efficiency, and geometric reasoning.

## Figures To Add

- **Figure 1.** End-to-end MOCKDOCK workflow: ChEMBL/PDB mining -> assay matching -> MCS derivation -> receptor preparation -> ligand preparation -> docking -> fragment RMSD validation -> normalized scoring and post hoc med-chem evaluation.
- **Figure 2.** Six-panel scatter plot of pChEMBL versus mean docking score across the five variance runs for the final benchmark tasks. Highlight the lower-quartile model-visible compounds.
- **Figure 3.** For each benchmark, show the 2D fragment/MCS and a 3D view of the fragment placement in the crystal structure pocket.
- **Figure 4.** Model comparison figure showing optimization trajectories or top-10 score distributions across targets.
- **Figure 5.** Example pose-validation figure comparing the crystal ligand and re-docked/reference-aligned fragment for a representative task.

## Data Availability

All benchmark data, configuration files, and code are available through the MOCKDOCK repository and packaged Python distribution. The release includes benchmark YAML files, curated bioactivity CSVs, pre-built receptor grids for the six final tasks, the `MDOracle` evaluation interface, and the scripts required to regenerate benchmark candidates from ChEMBL and the PDB.

## Supplementary Information

The main text should stay focused on benchmark motivation, core methodology, the six final tasks, and headline model results. The following details are better placed in supplementary material:

1. Full ligand-preparation details, including pH range, state caps, ETKDG settings, MMFF94s/UFF fallback logic, and PDBQT conversion details.
2. Complete top-30 variance analysis tables and per-system plots.
3. Full YAML schema and reproducibility details for benchmark configuration files.
4. Software versions, hardware details, and exact environment specifications.
5. Expanded per-target model tables and additional qualitative examples.

## References

To finalize before arXiv submission:

- Add canonical citations for GuacaMol, MOSES, MolOpt / PMO, DOCKSTRING, smina-docking-benchmark, MolScore, REINVENT, Lib-INVENT, AceGen, and PrexSyn.
- Add citations on QSAR reward hacking / applicability-domain failure, fragment-based drug discovery, and pose-validation best practices.
- Replace any remaining shorthand citation notes with the final BibTeX-backed references used in the manuscript.