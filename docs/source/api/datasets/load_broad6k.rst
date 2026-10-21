load_broad6k
============

Broad6k — bioactive molecules.

.. code-block:: python

   from torch_molecule.datasets import load_broad6k

Reference
---------

.. autofunction:: torch_molecule.datasets.load_broad6k

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_broad6k

   data = load_broad6k(target_cols=["220_692", "221_693"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
