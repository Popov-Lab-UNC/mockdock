Benchmark Reference & Guide
===========================

This section contains in-depth documentation on benchmark designs, scoring mechanics, custom target creation, and supporting automation scripts.

.. toctree::
   :maxdepth: 2

   benchmarks
   scoring
   custom_benchmark
   scripts

Summary of Topics
-----------------

* :doc:`benchmarks`: Detailed breakdown of the seven curated benchmark targets (CHK1, DPP4, ITK, PEPCK, PptT, TTK, VEGFR2), including crystal structures, resolution, ChEMBL assay IDs, and fragment definitions.
* :doc:`scoring`: The 5-step scoring pipeline from 2D substructure filtering to 3D docking, pose RMSD validation, score normalization, and reward clipping.
* :doc:`custom_benchmark`: Step-by-step protocol for adding, calibrating, and registering new target systems in **mockdock**.
* :doc:`scripts`: Tooling for multi-seed variance calibration, large-scale benchmarking, and figure generation.
