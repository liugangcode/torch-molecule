load_admet
==========

ADMET — pharmacokinetic and toxicity properties.

.. code-block:: python

   from torch_molecule.datasets import load_admet

Reference
---------

.. autofunction:: torch_molecule.datasets.load_admet

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_admet

   data = load_admet(target_cols=["AMES", "BBB_Martins"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
