Installation
============

Requirements
------------

* Python **>= 3.11**
* Linux or macOS
* **AutoDock-GPU v1.6** (Required for standard benchmark scoring; NVIDIA GPU recommended)

Step 1: Install AutoDock-GPU (Required, v1.6)
---------------------------------------------

**mockdock** uses `AutoDock-GPU <https://github.com/ccsb-scripps/AutoDock-GPU>`_ (tested with **v1.6**) as its primary docking and scoring backend.

Download a pre-compiled binary directly from `GitHub Releases <https://github.com/ccsb-scripps/AutoDock-GPU/releases>`_ or compile from source following the instructions in the `AutoDock-GPU repository <https://github.com/ccsb-scripps/AutoDock-GPU>`_.

Custom AutoDock-GPU Executable Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, **mockdock** searches for an ``adgpu`` executable on your system ``PATH``. You can specify a custom binary location using either:

1. The ``ADGPU_EXECUTABLE`` environment variable:

.. code-block:: bash

   export ADGPU_EXECUTABLE=/custom/path/to/adgpu

2. Or the ``adgpu_executable`` parameter in Python:

.. code-block:: python

   from mockdock import MDOracle

   oracle = MDOracle("CHK1", adgpu_executable="/custom/path/to/adgpu")

Step 2: Install mockdock
------------------------

Install **mockdock** with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/Popov-Lab-UNC/mockdock.git
   cd mockdock
   pip install -e ".[all]"

Or using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   git clone https://github.com/Popov-Lab-UNC/mockdock.git
   cd mockdock
   uv venv
   uv sync --all-extras

Optional Backends & Extras
--------------------------

AutoDock Vina Backend (CPU Fallback)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable CPU-based docking via AutoDock Vina (no GPU required):

.. code-block:: bash

   pip install -e ".[vina]"

Receptor Preparation & Custom Benchmark Tools
---------------------------------------------

For using standard benchmark targets, pre-computed AutoGrid maps are included and no extra tools are needed.

If you are creating **new custom benchmark systems** from raw PDB or CIF files:

.. code-block:: bash

   # Install ProDy for receptor splitting and cleaning
   pip install -e ".[receptor]"

   # (Optional) Install CCTBX for complex bond-order assignment
   conda install -c conda-forge cctbx-base

Building Documentation Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build these Shibuya Sphinx docs locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html

Open ``docs/_build/html/index.html`` in your browser.

Cache & Scratch Directory
-------------------------

**mockdock** caches global resources (such as downloaded ChEMBL bioactivity datasets and pre-computed AutoGrid files) in a persistent directory.

By default, this is ``~/.mockdock/``. You can override this path by setting the ``MOCKDOCK_CACHE_DIR`` environment variable:

.. code-block:: bash

   export MOCKDOCK_CACHE_DIR=/path/to/shared/mockdock_cache
