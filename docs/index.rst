mockdock documentation
======================

**mockdock** is a docking-based benchmarking package for chemical language models (CLMs) and generative algorithms performing fragment-constrained molecular generation.

.. image:: ../assets/Mock_Dock_Duck.svg
   :alt: MockDock banner
   :align: center
   :width: 85%

Overview
--------

Each benchmark system in **mockdock** is built around a curated protein–ligand crystal structure from the PDB paired with bioactivity-annotated reference compounds (mainly from ChEMBL).
Generative models are evaluated on their ability to grow or decorate a fixed 2D core fragment into high-scoring molecules while maintaining a similar 3D binding pose as the reference ligand.

.. image:: ../assets/MOCKDOCK.svg
   :alt: MOCKDOCK Benchmark Construction and Generative Model Evaluation Workflow
   :align: center
   :width: 100%

What mockdock provides
----------------------

* **Curated Target Benchmarks**: Seven protein targets (CHK1, DPP4, ITK, PEPCK, PptT, TTK, VEGFR2) with pre-computed AutoGrid maps and bioactivity baselines.
* **Standardized Oracle Interface**: :class:`~mockdock.MDOracle` handles SMILES sanitization, conformer generation, docking execution, pose RMSD validation, and score normalization.
* **Docking Backends**: **AutoDock-GPU** (GPU) and **AutoDock Vina** (CPU).
* **Post-hoc Evaluation**: :class:`~mockdock.MDEvaluator` calculates standardized metrics covering generation quality, medicinal chemistry alerts, and oracle call efficiency.

At a glance
-----------

Score molecules using a unified interface:

.. code-block:: python

   from mockdock import MDOracle

   # Instantiate oracle for a specific benchmark target
   oracle = MDOracle("CHK1", budget=1000, run_dir="./my_run")

   # Initial seed compounds (lowest-quartile bioactivity)
   initial_df = oracle.get_initial_compounds()

   # Substructure fragment constraint that generated molecules must contain
   fragment_smiles = oracle.fragment_smiles

   # Score candidate molecules (returns dict of {smiles: reward_score})
   scores = oracle.score(["CCO", "c1ccccc1"])

   # Inspect session history
   # (Results are automatically written to oracle.run_dir / "results.csv")
   print(oracle.results_df)

   # Inspect remaining budget
   print(oracle.budget_remaining)

   # Or export explicitly:
   oracle.results_df.write_csv("my_run/results.csv")

Computing benchmarking metrics after a run:

.. code-block:: python

   from mockdock import MDEvaluator

   evaluator = MDEvaluator("CHK1")
   metrics = evaluator.compute_metrics("my_run/results.csv")

   print(f"Top 10 Mean Score: {metrics['avg_top_10']:.3f}")
   print(f"MedChem Pass Fraction: {metrics['fraction_medchem_pass']:.1%}")

Documentation Sections
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Description
   * - :doc:`installation`
     - System requirements, AutoDock-GPU binary setup, CPU Vina fallback, and environment configuration.
   * - :doc:`running`
     - How to initialize oracles, query fragment constraints, batch score candidates, inspect session states, and connect with generative models.
   * - :doc:`evaluation`
     - How to run :class:`~mockdock.MDEvaluator`, full breakdown of all 22 metrics, and multi-model aggregate analysis.
   * - :doc:`reference/index`
     - Complete reference on the 7 standard targets, scoring equations, creating custom targets, and automation scripts.
   * - :doc:`api/index`
     - Python API reference for all public classes, methods, and modules.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   running
   evaluation
   reference/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
