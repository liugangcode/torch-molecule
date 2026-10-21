load_zinc250k
=============

ZINC250k — unlabeled molecules.

.. code-block:: python

   from torch_molecule.datasets import load_zinc250k

Reference
---------

.. autofunction:: torch_molecule.datasets.load_zinc250k

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_zinc250k

   data = load_zinc250k()
   print(len(data.data))
   print(data.target)  # None: this dataset has no labels.

See :doc:`../../dataset` for caching, custom data, and train/test splits.
