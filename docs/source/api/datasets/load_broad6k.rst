load_broad6k
============

Broad6k — bioactive molecules. Deprecated: this function calls ``load_dataset`` and will be removed.

Use instead
-----------

.. code-block:: python

   from torch_molecule.datasets import load_dataset

   data = load_dataset("broad6k")

Reference
---------

.. autofunction:: torch_molecule.datasets.load_broad6k

See :doc:`../../dataset` for caching, custom data, and train/test splits.
