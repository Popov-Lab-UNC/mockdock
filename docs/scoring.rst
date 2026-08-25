Scoring & Constraints
=====================

**mockdock** converts raw docking energies into standardized, bounded reward signals designed for reinforcement learning, genetic algorithms, and chemical language models.

The Scoring Pipeline
--------------------

When a molecule is submitted to :meth:`~mockdock.MDOracle.score`, it passes through a 5-step evaluation pipeline:

.. code-block:: text

   Candidate SMILES
          │
          ▼
   1. 2D Substructure Match? ──[No]──► Reward = 0.0
          │ [Yes]
          ▼
   2. 3D Conformer Prep & Docking (AD-GPU or Vina)
          │
          ▼
   3. 3D Pose RMSD <= Threshold (2.0 Å)? ──[No]──► Reward = 0.0
          │ [Yes]
          ▼
   4. Min-Max Normalization -> norm_score
          │
          ▼
   5. Reward Clipping [0.0, 1.0] -> reward_score

1. 2D Substructure Check
^^^^^^^^^^^^^^^^^^^^^^^^

The candidate molecule must contain the target's core fragment as a substructure (using RDKit's ``mol.HasSubstructMatch``).

If the fragment is missing, the molecule is assigned a reward score of ``0.0`` immediately without wasting GPU or CPU cycles on docking.

2. 3D Conformer Prep & Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* RDKit generates a 3D conformer with energy minimization (ETKDG / UFF).
* Meeko generates the PDBQT representation with assigned partial charges.
* The docking engine (AutoDock-GPU or AutoDock Vina) runs against the target's pre-computed grid maps.
* The lowest predicted binding energy (kcal/mol) is recorded as ``docking_score`` (more negative = stronger binding).

3. 3D Pose RMSD Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^

The docked pose of the candidate must maintain the experimental binding mode of the crystal fragment.

* The coordinates of the matched substructure atoms in the docked pose are compared to the reference crystal ligand fragment atoms.
* If the heavy-atom RMSD exceeds the threshold (default: ``2.0 Å``), the pose is considered biologically invalid and given a reward of ``0.0``.

4. Score Normalization
^^^^^^^^^^^^^^^^^^^^^^

Raw docking energies are normalized against target-specific calibration bounds:

.. math::

   \text{norm\_score} = \frac{\text{docking\_score} - \text{low\_score}}{\text{high\_score} - \text{low\_score}}

Where:
* ``low_score`` corresponds to the worst average docking score in the baseline ChEMBL dataset (mapped to 0.0).
* ``high_score`` corresponds to the best average docking score in the baseline ChEMBL dataset (mapped to 1.0).

5. Reward Clipping
^^^^^^^^^^^^^^^^^^

By default (``clip_reward_upper_bound = True``):

.. math::

   \text{reward\_score} = \min(\max(\text{norm\_score}, 0.0), 1.0)

This ensures rewards remain strictly within ``[0.0, 1.0]`` for reinforcement learning stability. Both ``norm_score`` (unclipped) and ``reward_score`` (clipped) are preserved in the results DataFrame.

Relaxing Constraints
--------------------

For unconstrained exploration or ablation studies, you can relax or disable specific filters:

.. code-block:: python

   oracle = MDOracle("CHK1")

   # Dock all molecules even if they lack the 2D fragment
   oracle._loader.require_fragment_match = False

   # Accept the best docking energy regardless of pose RMSD
   oracle._loader.require_pose_rmsd = False

   # Disable reward upper bound clipping
   oracle.clip_reward_upper_bound = False
