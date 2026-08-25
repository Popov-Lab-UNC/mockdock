Quickstart
==========

The primary entry point for integrating **mockdock** into a generative model training or evaluation loop is :class:`~mockdock.MDOracle`.

Basic Workflow
--------------

1. Listing Benchmarks
^^^^^^^^^^^^^^^^^^^^^

You can query all available benchmark systems:

.. code-block:: python

   from mockdock import MDOracle

   benchmarks = MDOracle.list_benchmarks()
   print(benchmarks)
   # ['CHK1', 'DPP4', 'ITK', 'PEPCK', 'PptT', 'TTK', 'VEGFR2']

2. Initializing the Oracle
^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate the oracle for your target system. You can specify the scoring budget, docking backend, and session output directory:

.. code-block:: python

   oracle = MDOracle(
       benchmark_name="CHK1",
       budget=1000,
       docking_backend="auto",  # Uses AutoDock-GPU if available, else Vina
       run_dir="./my_experiment_run",
   )

3. Getting Initial Starting Compounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generative algorithms (e.g. reinforcement learning, genetic algorithms, or fragment decorators) can use the low-affinity seed compounds from ChEMBL as starting points:

.. code-block:: python

   # Returns a Polars DataFrame of compounds in the bottom quartile of bioactivity
   initial_df = oracle.get_initial_compounds()
   print(initial_df.head())

4. Querying Fragment Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All generated molecules are constrained to contain the target's core fragment:

.. code-block:: python

   print("Required core fragment:", oracle.fragment_smiles)

5. Scoring Molecule Batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a list of candidate SMILES strings to :meth:`~mockdock.MDOracle.score`:

.. code-block:: python

   smiles_batch = [
       "Nc1ncnc2c1c(c[nH]2)c3ccc(NC(=O)N4CCNCC4)cc3",
       "COc1ccc2[nH]c(c(C#N)c2c1)c3cncc(N)n3",
       "CCO",  # Will fail 2D fragment match -> reward = 0.0
   ]

   rewards = oracle.score(smiles_batch)
   print(rewards)
   # {'Nc1ncnc2c1c(c[nH]2)c3ccc(NC(=O)N4CCNCC4)cc3': 0.824, ...}

6. Inspecting Results & Session State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During scoring, **mockdock** automatically flushes the cumulative results to ``results.csv`` inside ``oracle.run_dir`` after every batch.

You can inspect the full DataFrame in Python or export it explicitly:

.. code-block:: python

   # Access results as a Polars DataFrame
   results_df = oracle.results_df
   print(results_df)

   # Path to the auto-saved live CSV
   csv_path = oracle.run_dir / "results.csv"
   print("Live results saved at:", csv_path)

   # Or export explicitly to a custom location
   oracle.results_df.write_csv("./my_experiment_results.csv")

   # Check remaining oracle calls
   print(f"Remaining budget: {oracle.budget_remaining} / {oracle.budget}")

Results DataFrame Schema
------------------------

The ``results_df`` records complete per-molecule scoring details:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``smiles``
     - String
     - Input canonicalized SMILES.
   * - ``has_substruct``
     - Boolean
     - Whether the molecule matched the required 2D fragment.
   * - ``docking_score``
     - Float
     - Raw predicted binding energy (kcal/mol; lower is better).
   * - ``pose_rmsd``
     - Float
     - 3D RMSD of the docked fragment vs the crystal fragment pose (Å).
   * - ``norm_score``
     - Float
     - Unbounded normalized score relative to target calibration bounds.
   * - ``reward_score``
     - Float
     - Bounded reward score in [0.0, 1.0].
   * - ``is_valid``
     - Boolean
     - Whether docking conformer generation and scoring succeeded.
   * - ``elapsed_time``
     - Float
     - Time taken to process the batch (seconds).
