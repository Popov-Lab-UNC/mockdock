Evaluating Results
==================

**mockdock** provides a standalone evaluation engine, :class:`~mockdock.MDEvaluator`, to compute standardized metrics across runs without requiring docking engines or GPU resources.

This page explains how to evaluate individual experiment runs, interpret the full suite of metrics, and aggregate multi-model benchmarking experiments.

Single Run Evaluation
---------------------

.. tip::

   ``results.csv`` is generated automatically by :class:`~mockdock.MDOracle` during scoring and saved in ``oracle.run_dir / "results.csv"``.

Python API
^^^^^^^^^^

.. code-block:: python

   from mockdock import MDEvaluator

   # Initialize evaluator with target benchmark configuration
   evaluator = MDEvaluator("CHK1")

   # Compute metrics dictionary from a session CSV
   metrics = evaluator.compute_metrics("my_run/results.csv")

   # Print key summary metrics
   print(f"Top 10 Mean Reward: {metrics['avg_top_10']:.3f}")
   print(f"MedChem Pass Fraction: {metrics['fraction_medchem_pass']:.1%}")
   print(f"Effective Yield Rate: {metrics['effective_yield_rate']:.1%}")
   print(f"Internal Diversity: {metrics['internal_diversity']:.3f}")

   # Save metrics to JSON
   evaluator.compute_metrics("my_run/results.csv", output_path="my_run/eval_metrics.json")

Command-Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate any session CSV directly from the terminal:

.. code-block:: bash

   python -m mockdock.evaluator my_run/results.csv \
     --benchmark CHK1 \
     --output my_run/eval_metrics.json

Standard Metric Suite
---------------------

:class:`~mockdock.MDEvaluator` computes 17 standardized metrics across four main categories:

1. Generative Quality & Diversity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluates the structural validity, deduplication, and chemical variety of generated candidates.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Metric
     - Description
   * - ``validity``
     - Fraction of generated SMILES strings that parse into chemically valid RDKit molecules.
   * - ``uniqueness``
     - Fraction of valid molecules that are structurally unique.
   * - ``internal_diversity``
     - Average pairwise Tanimoto distance (using Morgan fingerprints, radius=2) among unique valid molecules.
   * - ``scaffold_diversity``
     - Fraction of unique Bemis-Murcko scaffolds among unique valid molecules.

2. Medicinal Chemistry & Drug-Likeness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies medicinal chemistry filters and property distributions to assess clinical/lead viability.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Metric
     - Description
   * - ``mean_qed``
     - Mean Quantitative Estimate of Drug-likeness (QED).
   * - ``mean_sa``
     - Mean Synthetic Accessibility score (1 = easily synthesizable, 10 = difficult).
   * - ``fraction_medchem_pass``
     - Fraction of unique valid fragment-containing molecules passing all structural alert filters (PAINS, BMS) and property rules.
   * - ``effective_medchem_pass``
     - Fraction of Effective Yield Rate molecules passing structural alerts (PAINS, BMS) and physchem bounds.

3. Novelty & Fragment Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measures whether generated candidates retain the required pharmacophore while exploring novel chemical space.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Metric
     - Description
   * - ``fragment_incorporation``
     - Fraction of unique valid molecules containing the required target fragment substructure.
   * - ``novelty``
     - Fraction of unique valid molecules not present in the starting seed dataset.
   * - ``snn``
     - Average maximum Tanimoto similarity to the nearest neighbor in the initial seed dataset.
   * - ``effective_yield_rate``
     - Fraction of all generated molecules that are valid, unique, contain the fragment, and are novel.

4. Optimization & Docking Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluates target-specific binding affinity optimization and docked pose validity.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Metric
     - Description
   * - ``avg_top_1`` / ``10`` / ``100``
     - Mean bounded reward score of the top-1, top-10, and top-100 unique molecules in the run.
   * - ``auc_top_10``
     - Area under the running top-10 reward trajectory across cumulative oracle calls (normalized to [0.0, 1.0]).
   * - ``valid_pose_rate``
     - Fraction of docked compounds whose docked fragment overlays the crystal structure within the RMSD threshold (≤ 2.0 Å).

Multi-Experiment Aggregation & Plotting
---------------------------------------

When running benchmark suites across multiple models, targets, and random seeds, you can aggregate all results and produce publication-ready comparison figures using ``scripts/analysis/analyze_experiments.py``.

.. code-block:: bash

   python scripts/analysis/analyze_experiments.py \
     --exps-dir exps/ \
     --output-dir analysis_results/

Generated Summary Artifacts:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``metrics_summary.csv``: Per-model, per-benchmark detailed evaluation table.
* ``metrics_summary_macro.csv``: Macro-averaged metrics across evaluated benchmark tasks.
* **Publication Figures**:
  * **Figure 1**: Generative Quality & Diversity metrics across models.
  * **Figure 2**: Optimization & Docking Performance (Top-10 reward, MedChem-filtered Top-10, AUC).
  * **Figure 3**: Medicinal chemistry distributions (QED, SA, Lipinski, structural alert pass rates).
  * **Figure 4**: Cumulative learning trajectory curves over oracle calls.
