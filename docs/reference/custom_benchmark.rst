Creating Custom Benchmarks
==========================

This guide explains how to define, curate, calibrate, and register a new target system in **mockdock**.

Step 1: Create the Benchmark TOML
---------------------------------

Add a new configuration file at ``src/mockdock/configs/<BenchmarkName>.toml``:

.. code-block:: toml

   benchmark_name = "MyTarget"
   pdb_id = "1ABC"
   target_id = "CHEMBL1234"
   doc_id = "CHEMBL5678"
   assay_id = "CHEMBL9012"
   ligand_resname = "LIG"
   fragment_smiles = "c1ccccc1"
   fragment_smiles_with_dummies = "*c1ccccc1"
   require_fragment_match = true
   require_pose_rmsd = true
   filter_during_optimization = true
   clip_reward_upper_bound = true
   low_score = -6.00     # Calibrated in Step 4
   high_score = -11.50   # Calibrated in Step 4
   rmsd_threshold = 2.0

Step 2: Curate Bioactivity Data
-------------------------------

Save the reference bioactivity dataset to ``src/mockdock/bioactivity_data/<BenchmarkName>.csv``.

Required columns:

* ``molecule_chembl_id``: ChEMBL compound identifier.
* ``canonical_smiles``: Standardized SMILES string.
* ``pchembl_value``: Experimental affinity measurement (:math:`-\log_{10} \text{IC}_{50}` / :math:`K_i`).

Step 3: Prepare Docking Grids & Crystal Ligand
----------------------------------------------

Place pre-computed AutoGrid files in ``src/mockdock/grids/<PDB_ID>/``:

* ``<PDB_ID>.maps.fld``: Grid definition field file.
* Associated atom map files (``.C.map``, ``.A.map``, ``.OA.map``, ``.e.map``, ``.d.map``, etc.).
* ``<PDB_ID>_ligand_corrected.sdf``: Cleaned crystal ligand SDF with correct bond orders.

Alternatively, if receptor tools are installed (``pip install -e ".[receptor]"``), **mockdock** can fetch and prepare the receptor PDB from the RCSB PDB using :class:`~mockdock.ReceptorPreparer`.

Step 4: 5-Fold Variance Calibration
-----------------------------------

Run the variance calibration script across 5 independent seeds to calibrate docking energy distributions and correlation with experimental bioactivity:

.. code-block:: bash

   python scripts/variance/run_variance.py \
     --config src/mockdock/configs/MyTarget.toml \
     --run-dir variance_runs/MyTarget \
     --output-dir variance_analysis/MyTarget \
     --n-iters 5

This generates:

* ``docking_vs_activity.png``: Pearson, Spearman, and :math:`R^2` correlation between docking scores and experimental :math:`\text{pChEMBL}` values.
* ``rmsd_distribution.png``: Distribution of fragment overlay RMSDs.
* Baseline energy statistics (:math:`\text{low\_score}` and :math:`\text{high\_score}`).

Step 5: Set Calibration Bounds
------------------------------

Update ``low_score`` and ``high_score`` in ``src/mockdock/configs/<BenchmarkName>.toml`` using the 5x variance calibration results. Once saved, the new target benchmark is immediately available via ``MDOracle("MyTarget")``.
