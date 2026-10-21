load_chembl2k
=============

ChEMBL2k — bioactive molecules.

.. code-block:: python

   from torch_molecule.datasets import load_chembl2k

Reference
---------

.. autofunction:: torch_molecule.datasets.load_chembl2k

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_chembl2k

   data = load_chembl2k(target_cols=["ABCB1", "EGFR"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
