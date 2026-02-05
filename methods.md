# Methods

## 1. Data Selection and Dataset Construction

### 1.1 Target and Structure Identification
The initial dataset was constructed by retrieving all human single-protein targets from the **ChEMBL36** database. For each identified target, associated crystal structures were retrieved from the **RCSB Protein Data Bank (PDB)** using UniProt accession IDs. Ligand information associated with these structures was simultaneously collected. To ensure the relevance of the benchmarking set, ligands were filtered to include only "drug-like" molecules, optimizing the selection for pharmaceutical relevance and structural quality.

### 1.2 Document Association and Compound Filtering
For each crystal ligand, we queried the ChEMBL database to identify associated documents, including patents and peer-reviewed publications. To ensure statistical robustness for each benchmark entry, we selected only those documents containing at least **20** measured compounds. Documents were matched to crystal structures based on the presence of the crystal ligand or close analogs, using **Morgan fingerprints** (radius 2, 2048 bits) and a **Tanimoto similarity threshold** of **0.7**. A similarity of **≥ 0.99** was used to definitively identify the crystal ligand within a document.

### 1.3 Benchmark Configuration Generation
For the selected compounds within each document, the **Maximum Common Substructure (MCS)** was calculated relative to the crystal ligand using the **FMCS (Find Maximum Common Substructure)** module in **RDKit**. To ensure representativeness, the MCS was computed across up to **100** compounds per document, requiring a **100% threshold** (the substructure must be present in all selected molecules). The computation enforced strict atom and bond comparison (matching elements and bond orders), with **complete rings** and **ring-to-ring matches** only. Benchmark configurations were then generated, each defined by a ChEMBL document ID, an RCSB PDB ID, and the MCS represented as a SMILES string.

## 2. Computational Workflow

### 2.1 Receptor Preparation
Crystal structures were downloaded in CIF format from the RCSB PDB. **ProDy** was employed to parse the CIF files and locate the primary instance of the crystal ligand. The ligand's 3D coordinates were used to identify and isolate protein chains within a **5.0 Å** distance of any ligand atom. The resulting receptor complex was exported as a PDB file.

The receptor was further refined using **Reduce2** to add hydrogens and optimize the hydrogen-bond network. Grid maps for docking were generated using **Meeko's** `mk_prepare_receptor` utility and **AutoGrid4**, applying a **5.0 Å** padding around the reference ligand to define the docking volume.

### 2.2 Ligand Preparation
Ligand data retrieved from ChEMBL included ChEMBL IDs, canonical SMILES, and bioactivity data (pChEMBL values). The ligands underwent a multi-step preparation pipeline:
1.  **Standardization:** **RDKit** was used to strip salts, normalize charges, and standardize the SMILES.
2.  **Tautomer and State Generation:** **MolScrub** was utilized to prepare appropriate tautomeric and ionization states within a pH range of **6.4 to 8.4**. A maximum of **16** distinct states were retained per ligand.
3.  **Conformer Generation:** 3D conformers were generated using the **ETKDGv3** algorithm in RDKit with small ring torsions enabled.
4.  **Docking Input Generation:** Final PDBQT files were prepared using **Meeko's** `mk_prepare_ligand`, with a total limit of **32** prepared states/conformers per unique compound.

### 2.3 Docking Campaign
A mass docking campaign was executed using **AutoDock-GPU** for high-throughput performance. For benchmarks utilizing **AutoDock Vina**, an exhaustiveness of **32** and a limit of **10** poses per run were specified. Docking was performed across all prepared benchmarks, targeting the binding site defined by the crystal ligand.

## 3. Analysis and Evaluation Metrics

### 3.1 Structural Validation
The accuracy of the docking poses was evaluated by calculating the **Root Mean Square Deviation (RMSD)** of the MCS between the docked ligand and the reference crystal ligand using RDKit. A threshold of **2.0 Å** was established as the criterion for a "passed" docking pose across all benchmarks.

### 3.2 Performance Metrics
Docking performance was assessed by correlating docking scores with experimental bioactivity (pChEMBL values). Several statistical metrics were calculated to filter and rank the results:
*   **Percent Passed:** The percentage of compounds meeting the 2.0 Å RMSD threshold.
*   **Coefficient of Determination ($R^2$)**
*   **Pearson Correlation Coefficient**
*   **Spearman's Rank Correlation Coefficient**

These metrics provide a comprehensive overview of both the structural accuracy and the scoring reliability of the docking workflow.
