Standard Benchmarks
===================

**mockdock** includes 7 curated protein-ligand benchmark targets with pre-computed AutoGrid maps and ChEMBL bioactivity calibrations.

Target Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 12 10 16 16 16 30

   * - Benchmark
     - PDB ID
     - Target ID
     - Reference Ligand
     - Calibration [Low, High]
     - Fragment SMILES
   * - **CHK1**
     - ``2R0U``
     - ``CHEMBL4630``
     - ``M54``
     - [-6.44, -11.79]
     - ``O=c1[nH]ccc2ccc3ccccc3c12``
   * - **DPP4**
     - ``2HHA``
     - ``CHEMBL284``
     - ``3TP``
     - [-6.21, -11.23]
     - ``Cc1nnc2n1CCN(c1ccccc1)C2``
   * - **ITK**
     - ``3QGW``
     - ``CHEMBL2959``
     - ``L7A``
     - [-6.55, -11.45]
     - ``Nc1ncnc2c1c(C(=O)N)c[nH]2``
   * - **PEPCK**
     - ``2GMV``
     - ``CHEMBL2911``
     - ``UN8``
     - [-6.12, -10.98]
     - ``O=C(O)c1c[nH]c2ccccc12``
   * - **PptT**
     - ``8GKF``
     - Custom
     - ``D16``
     - [-6.30, -11.50]
     - ``CC1=NC(c2c(N1)ccc([*])c2)=O``
   * - **TTK**
     - ``3WZJ``
     - ``CHEMBL3983``
     - ``O43``
     - [-6.48, -11.62]
     - ``Nc1nccc(n1)-c1cccnc1``
   * - **VEGFR2**
     - ``3VHE``
     - ``CHEMBL279``
     - ``42Q``
     - [-6.70, -12.10]
     - ``c1cnc2[nH]ccc2c1``

Benchmark System Details
------------------------

CHK1 (Checkpoint Kinase 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2R0U``
* **Resolution**: 1.70 Å
* **Description**: Serine/threonine-protein kinase involved in checkpoint-mediated cell cycle arrest. The benchmark fragment corresponds to a polycyclic lactam core from compound M54.

DPP4 (Dipeptidyl Peptidase IV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2HHA``
* **Resolution**: 2.05 Å
* **Description**: Protease target for type 2 diabetes. Uses a piperazine-fused triazole scaffold.

ITK (Interleukin-2 Inducible T-cell Kinase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3QGW``
* **Resolution**: 2.00 Å
* **Description**: Tyrosine kinase key for T-cell signaling. Features an amino-pyrimidine / pyrrolopyrimidine hinge-binding motif.

PEPCK (Phosphoenolpyruvate Carboxykinase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``2GMV``
* **Resolution**: 1.50 Å
* **Description**: Key metabolic enzyme in gluconeogenesis. The fragment is an indole-2-carboxylic acid scaffold.

PptT (Phosphopantetheinyl Transferase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``8GKF``
* **Resolution**: 1.85 Å
* **Description**: Essential *Mycobacterium tuberculosis* enzyme target. Uses a quinazolinone / imidazopyridine derived fragment.

TTK (Dual Specificity Protein Kinase TTK / MPS1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3WZJ``
* **Resolution**: 2.10 Å
* **Description**: Essential mitotic spindle assembly kinase. Features a bipyridine-like motif.

VEGFR2 (Vascular Endothelial Growth Factor Receptor 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **PDB ID**: ``3VHE``
* **Resolution**: 1.55 Å
* **Description**: Receptor tyrosine kinase governing angiogenesis. Uses a pyrrolopyridine hinge binder core.

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
