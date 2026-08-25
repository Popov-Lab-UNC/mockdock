Installation
============

Requirements
------------

* Python **>= 3.11**
* Linux or macOS
* (Optional) NVIDIA GPU with CUDA for AutoDock-GPU acceleration

Standard Installation
---------------------

Clone the repository and install in editable mode with ``pip``:

.. code-block:: bash

   git clone https://github.com/Popov-Lab-UNC/mockdock.git
   cd mockdock
   pip install -e .

Or using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   git clone https://github.com/Popov-Lab-UNC/mockdock.git
   cd mockdock
   uv venv
   uv sync

Optional Backends
-----------------

AutoDock Vina Backend
^^^^^^^^^^^^^^^^^^^^^

To enable CPU-based docking via AutoDock Vina:

.. code-block:: bash

   pip install -e ".[vina]"

Receptor Preparation Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are creating custom benchmarks and need to prepare receptors from raw PDB files rather than using pre-computed AutoGrid maps:

.. code-block:: bash

   pip install -e ".[receptor]"

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
