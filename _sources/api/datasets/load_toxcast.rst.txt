load_toxcast
============

ToxCast — toxicity assays. Deprecated: this function calls ``load_dataset`` and will be removed.

Use instead
-----------

.. code-block:: python

   from torch_molecule.datasets import load_dataset

   data = load_dataset("toxcast")

Reference
---------

.. autofunction:: torch_molecule.datasets.load_toxcast

See :doc:`../../dataset` for caching, custom data, and train/test splits.
