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

.. image:: ../assets/MOCKDOCK.pdf
   :alt: MOCKDOCK Benchmark Construction and Generative Model Evaluation Workflow
   :align: center
   :width: 100%

What mockdock provides
----------------------

* **7 Curated Target Benchmarks**: Clinically relevant kinases and enzyme targets (CHK1, DPP4, ITK, PEPCK, PptT, TTK, VEGFR2) with pre-computed AutoGrid maps and ChEMBL baselines.
* **Standardized Oracle Interface**: :class:`~mockdock.MDOracle` handles SMILES sanitization, conformer generation, docking execution, pose RMSD validation, and score normalization.
* **Dual Docking Backends**: Seamless support for hardware-accelerated **AutoDock-GPU** (GPU) and **AutoDock Vina** (CPU/cross-platform).
* **Comprehensive Post-hoc Evaluation**: :class:`~mockdock.MDEvaluator` calculates 19+ standardized metrics covering generative quality, medicinal chemistry alerts (PAINS, BMS, Lipinski), novelty, top-10 optimization performance, and oracle call efficiency.
* **Reproducible Calibration Workflow**: Protocol and scripts for setting up new custom protein targets with 5-fold variance calibration.

At a glance
-----------

Scoring molecules during generative model training / reinforcement learning:

.. code-block:: python

   from mockdock import MDOracle

   # Instantiate oracle for a specific benchmark target
   oracle = MDOracle("CHK1", budget=1000, run_dir="./my_run")

   # Initial seed compounds (lowest-quartile ChEMBL bioactivity)
   initial_df = oracle.get_initial_compounds()

   # Substructure fragment constraint that generated molecules must contain
   fragment_smiles = oracle.fragment_smiles

   # Score candidate molecules (returns dict of {smiles: reward_score})
   scores = oracle.score(["CCO", "c1ccccc1"])

   # Inspect session history & remaining budget
   # (Results are automatically written to oracle.run_dir / "results.csv")
   print(oracle.results_df)
   print(oracle.budget_remaining)

   # Or export explicitly:
   oracle.results_df.write_csv("my_run/results.csv")

Computing comprehensive benchmarking metrics after a run:

.. code-block:: python

   from mockdock import MDEvaluator

   evaluator = MDEvaluator("CHK1")
   metrics = evaluator.compute_metrics("my_run/results.csv")

   print(f"Top 10 Mean Score: {metrics['avg_top_10']:.3f}")
   print(f"MedChem Pass Fraction: {metrics['fraction_medchem_pass']:.1%}")

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   benchmarks
   scoring
   evaluation
   custom_benchmark
   scripts

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
