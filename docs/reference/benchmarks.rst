Benchmark Tasks
===============

**mockdock** provides a curated suite of structure-grounded benchmark tasks for evaluating fragment-constrained molecular generation. Each benchmark task pairs a high-resolution protein–ligand crystal structure with a bioactivity-annotated chemical series and an experimental Maximum Common Substructure (MCS) constraint.

The framework is designed to be extensible: models can be evaluated on the standard pre-configured tasks below or on custom user-defined targets calibrated with the automated pipeline (see :doc:`custom_benchmark`).

Fragment Constraints (MCS Anchors)
----------------------------------

In structure-based prospective design, generative algorithms are tasked with expanding or decorating a validated chemical hit while preserving its core binding mode. **mockdock** anchors generation to a conserved core fragment—the Maximum Common Substructure (MCS) extracted from the crystallographic series.

This constraint serves two key purposes:

1. **Practical Hit Expansion**: Models are evaluated on their ability to grow functional groups from defined exit vectors, reflecting real-world medicinal chemistry campaigns.
2. **Quantitative 3D Pose Validation**: The co-crystallized fragment coordinates provide ground-truth atomic positions. A candidate molecule is only rewarded if its docked fragment satisfies a stringent structural constraint (:math:`\le 2.0\text{ \AA}` heavy-atom RMSD relative to crystal coordinates).

.. image:: ../../assets/figure2_fragments.png
   :alt: Chemical structures of the fragment constraints for each MOCKDOCK task
   :align: center
   :width: 85%

*Figure: Chemical structures of the conserved MCS fragment constraints across benchmark tasks, showing defined exit vectors (attachment points) and core scaffolds.*

Task Overview
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 12 10 8 12 14 14 20

   * - Benchmark
     - Target ID
     - PDB ID
     - Resolution
     - Reference Ligand
     - Spearman :math:`\rho`
     - % Passed RMSD
     - Calibration [Low, High]
   * - **CHK1**
     - ``CHEMBL4630``
     - ``2R0U``
     - 1.70 Å
     - ``M54``
     - :math:`-0.83 \pm 0.01`
     - :math:`76.8\% \pm 2.7\%`
     - [-6.44, -11.79]
   * - **DPP4**
     - ``CHEMBL284``
     - ``2HHA``
     - 2.05 Å
     - ``3TP``
     - :math:`-0.86 \pm 0.02`
     - :math:`100.0\% \pm 0.0\%`
     - [-6.21, -11.23]
   * - **ITK**
     - ``CHEMBL2959``
     - ``3QGW``
     - 2.00 Å
     - ``L7A``
     - :math:`-0.83 \pm 0.03`
     - :math:`99.1\% \pm 1.8\%`
     - [-6.55, -11.45]
   * - **PEPCK**
     - ``CHEMBL2911``
     - ``2GMV``
     - 1.50 Å
     - ``UN8``
     - :math:`-0.80 \pm 0.02`
     - :math:`100.0\% \pm 0.0\%`
     - [-6.12, -10.98]
   * - **PptT**
     - ``CHEMBL5465373``
     - ``8GKF``
     - 1.85 Å
     - ``D16``
     - :math:`-0.74 \pm 0.02`
     - :math:`93.0\% \pm 1.6\%`
     - [-6.30, -11.50]
   * - **TTK**
     - ``CHEMBL3983``
     - ``3WZJ``
     - 2.10 Å
     - ``O43``
     - :math:`-0.78 \pm 0.01`
     - :math:`91.2\% \pm 4.7\%`
     - [-6.48, -11.62]
   * - **VEGFR2**
     - ``CHEMBL279``
     - ``3VHE``
     - 1.55 Å
     - ``42Q``
     - :math:`-0.87 \pm 0.00`
     - :math:`100.0\% \pm 0.0\%`
     - [-6.70, -12.10]

.. note::
   Spearman :math:`\rho` and RMSD pass rates reflect mean :math:`\pm` standard deviation across 5 independent replicate calibration runs on the reference series. Six tasks were derived directly from ChEMBL; PptT was curated from a published *M. tuberculosis* inhibitor series.

Benchmark System Details
------------------------

CHK1 (Checkpoint Kinase 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2R0U``
* **Target ID**: ``CHEMBL4630``
* **Resolution**: 1.70 Å
* **Fragment SMILES**: ``O=c1[nH]ccc2ccc3ccccc3c12``
* **Description**: Serine/threonine-protein kinase involved in checkpoint-mediated cell cycle arrest. The benchmark fragment corresponds to a polycyclic lactam core from co-crystallized compound M54. Docking scores correlate strongly with molecular weight in this series.

DPP4 (Dipeptidyl Peptidase IV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2HHA``
* **Target ID**: ``CHEMBL284``
* **Resolution**: 2.05 Å
* **Fragment SMILES**: ``Cc1nnc2n1CCN(c1ccccc1)C2``
* **Description**: Serine exopeptidase and established target for type 2 diabetes. Uses a piperazine-fused triazole scaffold with 100% baseline RMSD adherence in the reference series.

ITK (Interleukin-2 Inducible T-cell Kinase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3QGW``
* **Target ID**: ``CHEMBL2959``
* **Resolution**: 2.00 Å
* **Fragment SMILES**: ``Nc1ncnc2c1c(C(=O)N)c[nH]2``
* **Description**: Non-receptor tyrosine kinase key for T-cell signaling. Features an amino-pyrimidine / pyrrolopyrimidine hinge-binding motif with excellent baseline RMSD stability (99.1%).

PEPCK (Phosphoenolpyruvate Carboxykinase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2GMV``
* **Target ID**: ``CHEMBL2911``
* **Resolution**: 1.50 Å
* **Fragment SMILES**: ``O=C(O)c1c[nH]c2ccccc12``
* **Description**: Key metabolic enzyme regulating gluconeogenesis. The fragment is an indole-2-carboxylic acid scaffold exhibiting high baseline pose reproducibility (100% passed RMSD).

PptT (Phosphopantetheinyl Transferase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``8GKF``
* **Target ID**: ``CHEMBL5465373``
* **Resolution**: 1.85 Å
* **Fragment SMILES**: ``CC1=NC(c2c(N1)ccc([*])c2)=O``
* **Description**: Essential *Mycobacterium tuberculosis* enzyme target curated from a published medicinal chemistry campaign. Notably, PptT is the sole task where potency is largely decoupled from molecular weight ($r = 0.45$), making it an important probe for models that optimize docking score via molecular weight inflation.

TTK (Dual Specificity Protein Kinase TTK / MPS1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3WZJ``
* **Target ID**: ``CHEMBL3983``
* **Resolution**: 2.10 Å
* **Fragment SMILES**: ``Nc1nccc(n1)-c1cccnc1``
* **Description**: Essential mitotic spindle assembly kinase. Features a flexible bipyridine-like motif with a broad dynamic score range.

VEGFR2 (Vascular Endothelial Growth Factor Receptor 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3VHE``
* **Target ID**: ``CHEMBL279``
* **Resolution**: 1.55 Å
* **Fragment SMILES**: ``c1cnc2[nH]ccc2c1``
* **Description**: Receptor tyrosine kinase governing tumor angiogenesis. Uses a compact pyrrolopyridine hinge binder that presents a challenging optimization landscape with stringent synthetic complexity trade-offs.

Special Scaffold Notes
----------------------

PptT / LibInvent Two-Attachment Scaffold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For single-exit-vector generative models (e.g. PromptSMILES, REINVENT), PptT uses the single-dummy fragment:

.. code-block:: text

   CC1=NC(c2c(N1)ccc([*])c2)=O

However, fragment decorators requiring two attachment points (such as LibInvent) must use the dedicated two-dummy scaffold defined in ``src/mockdock/configs/PptT.toml``:

.. code-block:: text

   O=C1N=C([*])Nc2c1cc([*])cc2

Ensure you specify the appropriate scaffold constraint in your model configuration when running multi-attachment benchmarks.
