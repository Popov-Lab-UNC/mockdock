Scripts & Automation
====================

The ``scripts/`` directory provides end-to-end automation for experiment execution, multi-model analysis, and target calibration.

Multi-Model Experiment Analysis
-------------------------------

``scripts/analysis/analyze_experiments.py`` aggregates results across multiple models, benchmarks, and random seeds:

.. code-block:: bash

   python scripts/analysis/analyze_experiments.py \
     --exps-dir exps/ \
     --output-dir analysis_results/

Generated outputs:

* ``metrics_summary.csv``: Per-model, per-benchmark detailed evaluation table.
* ``metrics_summary_macro.csv``: Macro-averaged metrics across all target benchmarks.
* **Publication Figures**:
  * **Figure 1**: Generative quality & diversity metrics.
  * **Figure 2**: Optimization & docking performance (top-10, AUC).
  * **Figure 3**: Medicinal chemistry & property filters.
  * **Figure 4**: Cumulative trajectory curves over oracle calls.

Variance & Baseline Calibration
-------------------------------

``scripts/variance/run_variance.py`` runs multi-seed docking calibration on ChEMBL compounds to determine target baseline energy bounds and validate scoring reliability:

.. code-block:: bash

   python scripts/variance/run_variance.py \
     --config src/mockdock/configs/CHK1.toml \
     --run-dir variance_runs/CHK1 \
     --output-dir variance_analysis/CHK1 \
     --n-iters 5

Docking Workflow Runner
-----------------------

``scripts/docking/run_workflow.py`` executes docking directly for given SMILES files or datasets without launching a full generative training loop:

.. code-block:: bash

   python scripts/docking/run_workflow.py \
     --benchmark CHK1 \
     --smiles-file input_molecules.smi \
     --output-dir docking_results/
