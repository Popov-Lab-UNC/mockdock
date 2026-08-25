Evaluation Metrics
==================

**mockdock** provides a standalone evaluation engine, :class:`~mockdock.MDEvaluator`, to compute standardized metrics across runs without needing docking engines or GPU resources.

Usage
-----

.. tip::

   ``results.csv`` is generated automatically by :class:`~mockdock.MDOracle` during scoring and saved in ``oracle.run_dir / "results.csv"``. You can also export custom DataFrames via ``oracle.results_df.write_csv("my_results.csv")``.

Python API
^^^^^^^^^^

.. code-block:: python

   from mockdock import MDEvaluator
   from pathlib import Path

   # Initialize evaluator with target benchmark configuration
   evaluator = MDEvaluator("CHK1")

   # Compute metrics from a session CSV (e.g. from oracle.run_dir / "results.csv")
   metrics = evaluator.compute_metrics(Path("my_run/results.csv"))

   # Save metrics to JSON
   evaluator.save_metrics(metrics, Path("my_run/eval_metrics.json"))

CLI Usage
^^^^^^^^^

.. code-block:: bash

   python -m mockdock.evaluator my_run/results.csv --benchmark CHK1 --output my_run/eval_metrics.json

Standard Metric Suite
---------------------

Generative Quality & Diversity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``validity``
     - Fraction of generated SMILES that parse into valid RDKit molecules.
   * - ``uniqueness``
     - Fraction of valid molecules that are structurally distinct.
   * - ``internal_diversity``
     - Average pairwise Tanimoto distance (Morgan fingerprints) among unique valid molecules.
   * - ``scaffold_diversity``
     - Fraction of unique Bemis-Murcko scaffolds among unique valid molecules.

Medicinal Chemistry & Drug-Likeness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``mean_qed``
     - Mean quantitative estimate of drug-likeness (QED).
   * - ``mean_sa``
     - Mean synthetic accessibility score (1 = easily synthesizable, 10 = difficult).
   * - ``fraction_lipinski``
     - Fraction of unique valid molecules satisfying all 4 Lipinski Rule-of-5 criteria.
   * - ``fraction_pains_free``
     - Fraction of unique valid molecules not triggering any PAINS alerts.
   * - ``fraction_bms_free``
     - Fraction of unique valid molecules not triggering any BMS structural alerts.
   * - ``fraction_medchem_pass``
     - Fraction passing both structural alerts (PAINS, BMS) and property filters.

Novelty & Fragment Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``fragment_incorporation``
     - Fraction of unique valid molecules containing the required target fragment substructure.
   * - ``novelty``
     - Fraction of unique valid molecules not present in the starting seed dataset.
   * - ``nonidenticality``
     - Fraction of generated molecules non-identical to their prompt/input parent.
   * - ``effective_novelty``
     - Fraction of molecules that are both novel and non-identical.
   * - ``snn``
     - Average maximum Tanimoto similarity to the initial seed dataset (Nearest Neighbor).
   * - ``effective_hit_rate``
     - Fraction of all generated molecules that are valid, unique, contain the fragment, and are novel.

Optimization & Docking Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``avg_top_1`` / ``10`` / ``100``
     - Mean bounded reward score of top-1, top-10, and top-100 scoring molecules.
   * - ``avg_top_1_norm`` / ``10`` / ``100``
     - Mean unbounded normalized score of top-1, top-10, and top-100 molecules.
   * - ``avg_top_10_filtered``
     - Mean reward score of the top-10 molecules that pass all MedChem filters.
   * - ``auc_top_10``
     - Area under the running top-10 reward trajectory across cumulative oracle calls.
   * - ``auc_top_10_filtered``
     - AUC of the running top-10 curve restricted to MedChem-passing compounds.
   * - ``valid_pose_rate``
     - Fraction of docked compounds whose docked fragment overlays the crystal within the RMSD threshold.
   * - ``oracle_efficiency_80``
     - Number of oracle calls required to reach 80% of the final top-10 reward score (lower = faster learning).
   * - ``oracle_efficiency_100``
     - Number of oracle calls required to reach a top-10 reward of 1.0.
