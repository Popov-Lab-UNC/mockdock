Benchmark Tasks
===============

**mockdock** provides a curated suite of structure-grounded benchmark tasks for evaluating fragment-constrained molecular generation. Each benchmark task pairs a high-resolution protein–ligand crystal structure with a bioactivity-annotated chemical series and an experimental Maximum Common Substructure (MCS) constraint.

The framework is designed to be extensible: models can be evaluated on the standard pre-configured tasks below or on custom user-defined targets calibrated with the automated pipeline (see :doc:`custom_benchmark`).

Fragment Constraints (MCS Anchors)
----------------------------------

In structure-based prospective design, generative algorithms are tasked with expanding or decorating a validated chemical hit while preserving its core binding mode. **mockdock** anchors generation to a conserved core fragment—the Maximum Common Substructure (MCS) extracted from the crystallographic series.

This constraint serves two key purposes:

1. **Practical Hit Expansion**: Models are evaluated on their ability to grow functional groups from defined exit vectors, reflecting real-world medicinal chemistry campaigns.
2. **Quantitative 3D Pose Validation**: The co-crystallized fragment coordinates provide ground-truth atomic positions. A candidate molecule is only rewarded if its docked fragment satisfies a stringent structural constraint (≤ 2.0 Å heavy-atom RMSD relative to crystal coordinates).

.. image:: ../../assets/figure2_fragments.png
   :alt: Chemical structures of the fragment constraints for each MOCKDOCK task
   :align: center
   :width: 85%

*Figure: Chemical structures of the conserved MCS fragment constraints across benchmark tasks, showing defined exit vectors (attachment points) and core scaffolds.*

Task Overview
-------------

.. list-table::
   :header-rows: 1
   :widths: 12 28 10 32 18

   * - Task
     - Target Protein
     - Structure ID
     - Fragment SMILES
     - Calibration [0.0 → 1.0]
   * - **CHK1**
     - Checkpoint Kinase 1
     - ``2R0U``
     - ``O=c1[nH]ccc2ccc3ccccc3c12``
     - [-6.44, -11.79]
   * - **DPP4**
     - Dipeptidyl Peptidase IV
     - ``2HHA``
     - ``Cc1nnc2n1CCN(c1ccccc1)C2``
     - [-6.21, -11.23]
   * - **ITK**
     - Interleukin-2 Inducible T-cell Kinase
     - ``3QGW``
     - ``Nc1ncnc2c1c(C(=O)N)c[nH]2``
     - [-6.55, -11.45]
   * - **PEPCK**
     - Phosphoenolpyruvate Carboxykinase
     - ``2GMV``
     - ``O=C(O)c1c[nH]c2ccccc12``
     - [-6.12, -10.98]
   * - **PptT**
     - Phosphopantetheinyl Transferase
     - ``8GKF``
     - ``CC1=NC(=O)c2ccccc2N1``
     - [-6.30, -11.50]
   * - **TTK**
     - Mitotic Kinase TTK / MPS1
     - ``3WZJ``
     - ``Nc1nccc(n1)-c1cccnc1``
     - [-6.48, -11.62]
   * - **VEGFR2**
     - Vascular Endothelial Growth Factor Receptor 2
     - ``3VHE``
     - ``c1cnc2[nH]ccc2c1``
     - [-6.70, -12.10]

.. note::
   For custom target systems, see :doc:`custom_benchmark` for step-by-step instructions on calibrating new AutoGrid maps and bioactivity thresholds. For extensive statistical replication analyses and cross-architecture benchmarking comparisons, please refer to the MOCKDOCK research paper.

Special Scaffold Constraints
----------------------------

PptT / LibINVENT Multi-Attachment Scaffold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For single-exit-vector generative models (e.g. PromptSMILES, REINVENT), PptT uses the default single-dummy fragment:

.. code-block:: text

   CC1=NC(c2c(N1)ccc([*])c2)=O

However, fragment decorators requiring two attachment points (such as LibINVENT) must use the dedicated two-dummy scaffold defined in ``src/mockdock/configs/PptT.toml``:

.. code-block:: text

   O=C1N=C([*])Nc2c1cc([*])cc2

Ensure you specify the appropriate scaffold constraint in your model configuration when running multi-attachment benchmarks.
