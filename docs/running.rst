Running Benchmarks
==================

The primary interface for running benchmarks and integrating **mockdock** into generative model workflows is :class:`~mockdock.MDOracle`.

This guide covers everything you need to initialize oracles, query benchmark properties, score molecule batches, inspect session history, and connect with generative models (reinforcement learning, genetic algorithms, chemical language models).

Overview
--------

.. code-block:: text

   ┌───────────────────────────────────────────────────────────┐
   │                     Your Model / Loop                     │
   │   (e.g., REINVENT, PromptSMILES, Genetic Algorithm, etc.)  │
   └───────────────┬───────────────────────────▲───────────────┘
                   │ Candidate SMILES          │ Reward Scores
                   ▼                           │ [0.0 - 1.0]
   ┌───────────────────────────────────────────┴───────────────┐
   │                    mockdock.MDOracle                      │
   │                                                           │
   │  1. 2D Substructure Filter                                │
   │  2. 3D Conformation Prep (RDKit / Meeko)                 │
   │  3. Docking Engine (AutoDock-GPU / Vina)                  │
   │  4. 3D Pose RMSD Constraint Check                         │
   │  5. Score Normalization & Clipping                        │
   │                                                           │
   │  ► Auto-saves live history to run_dir/results.csv         │
   └───────────────────────────────────────────────────────────┘

1. Listing Available Benchmarks
-------------------------------

You can inspect the list of available curated target systems directly in Python:

.. code-block:: python

   from mockdock import MDOracle

   benchmarks = MDOracle.list_benchmarks()
   print(benchmarks)
   # ['CHK1', 'DPP4', 'ITK', 'PEPCK', 'PptT', 'TTK', 'VEGFR2']

To learn more about the biological background, active fragments, and PDB structures for each target, see :doc:`reference/benchmarks`.

2. Initializing the Oracle
--------------------------

Instantiate :class:`~mockdock.MDOracle` for your target of choice:

.. code-block:: python

   from mockdock import MDOracle

   oracle = MDOracle(
       benchmark_name="CHK1",
       budget=1000,             # Total allowed molecule scoring calls
       docking_backend="auto",   # Uses AutoDock-GPU if available, falls back to Vina
       run_dir="./my_run",       # Directory where results.csv and logs are stored
   )

Configuration Parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``benchmark_name`` (*str*): The name of the target benchmark (e.g. ``"CHK1"``, ``"VEGFR2"``).
* ``budget`` (*int*, default: ``1000``): Total scoring budget. Once exhausted, calls return cached or zero rewards.
* ``docking_backend`` (*str*, default: ``"auto"``): Docking engine to use:
  * ``"auto"``: Automatically selects AutoDock-GPU if detected on ``PATH`` or ``ADGPU_EXECUTABLE``, otherwise falls back to AutoDock Vina.
  * ``"adgpu"``: Explicitly require AutoDock-GPU.
  * ``"vina"``: Explicitly require AutoDock Vina (CPU).
* ``run_dir`` (*str | Path*, optional): Output directory for the session. Live results are written to ``<run_dir>/results.csv`` after every batch.
* ``adgpu_executable`` (*str | Path*, optional): Custom path to the ``adgpu`` binary.

3. Accessing Initial Starting Compounds
---------------------------------------

Many generative workflows (e.g. fragment decorators, genetic algorithms, reinforcement learning) require seed molecules with known low-to-moderate affinity. **mockdock** provides baseline compounds curated from ChEMBL:

.. code-block:: python

   # Returns a Polars DataFrame of compounds in the bottom quartile of bioactivity
   initial_df = oracle.get_initial_compounds()

   print(initial_df.head())
   # Columns: ['molecule_chembl_id', 'canonical_smiles', 'pchembl_value']

4. Querying Fragment Constraints
--------------------------------

All generated molecules are required to contain the target's core fragment. You can access the 2D SMARTS / SMILES pattern required by the target:

.. code-block:: python

   # Canonical SMILES of the target core fragment
   print("Required fragment:", oracle.fragment_smiles)

   # Fragment with dummy attachment points (if applicable)
   print("Fragment with dummies:", oracle.fragment_smiles_with_dummies)

5. Scoring Molecule Batches
---------------------------

Pass candidate SMILES strings (as a list or sequence) to :meth:`~mockdock.MDOracle.score`:

.. code-block:: python

   smiles_batch = [
       "Nc1ncnc2c1c(c[nH]2)c3ccc(NC(=O)N4CCNCC4)cc3",
       "COc1ccc2[nH]c(c(C#N)c2c1)c3cncc(N)n3",
       "CCO",  # Lacks the core fragment -> fails 2D check -> reward = 0.0
   ]

   # Returns a dict mapping {smiles: reward_score}
   rewards = oracle.score(smiles_batch)

   for smi, reward in rewards.items():
       print(f"{smi} -> Reward: {reward:.4f}")

Scoring Behavior:
^^^^^^^^^^^^^^^^^

* **Missing Substructure**: Assigned ``0.0`` immediately without running conformer generation or docking.
* **Invalid SMILES / Chemistry**: Assigned ``0.0`` and marked as invalid.
* **Unfavorable / Broken 3D Pose (RMSD > 2.0 Å)**: Assigned ``0.0``.
* **Valid Poses**: Assigned a normalized, bounded reward in ``[0.0, 1.0]``.
* **Deduplication & Caching**: If a SMILES string was scored previously in the same session, its cached result is returned without consuming additional budget or docking computation.

6. Inspecting Session Results & State
-------------------------------------

During scoring, cumulative results are automatically flushed to ``results.csv`` inside ``oracle.run_dir`` after every batch.

You can also access the results in Python at any point:

.. code-block:: python

   # Access results as a Polars DataFrame
   df = oracle.results_df
   print(df)

   # Check budget status
   print(f"Calls made: {oracle.budget - oracle.budget_remaining} / {oracle.budget}")
   print(f"Budget remaining: {oracle.budget_remaining}")

   # Explicitly export to a custom CSV path if needed
   oracle.results_df.write_csv("my_custom_results.csv")

Results DataFrame Schema
------------------------

The generated ``results.csv`` records detailed per-molecule evaluations:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Column
     - Type
     - Description
   * - ``smiles``
     - String
     - Input canonicalized SMILES.
   * - ``has_substruct``
     - Boolean
     - Whether the molecule matched the target's 2D fragment constraint.
   * - ``docking_score``
     - Float
     - Raw predicted binding energy from the docking engine (kcal/mol; lower is better).
   * - ``pose_rmsd``
     - Float
     - 3D heavy-atom RMSD of the docked fragment compared to the crystal fragment pose (Å).
   * - ``norm_score``
     - Float
     - Unbounded normalized score relative to target calibration bounds.
   * - ``reward_score``
     - Float
     - Bounded reward score clipped to ``[0.0, 1.0]``.
   * - ``is_valid``
     - Boolean
     - Whether 3D preparation, docking, and scoring completed successfully.
   * - ``elapsed_time``
     - Float
     - Wall-clock time taken to process the molecule (seconds).

Standalone Docking via CLI
--------------------------

If you want to run docking directly on a predefined list of SMILES without writing a Python script:

.. code-block:: bash

   python scripts/docking/run_workflow.py \
     --benchmark CHK1 \
     --smiles-file input_molecules.smi \
     --output-dir docking_results/

Next Steps
----------

* **Evaluate your benchmark run**: Proceed to :doc:`evaluation` to compute standardized metrics across generative quality, medicinal chemistry alerts, and docking optimization.
* **Explore target specifics**: Check :doc:`reference/benchmarks` for details on all 7 systems.
* **Understand the scoring formula**: Read :doc:`reference/scoring` to learn about RMSD constraints and min-max normalization.
