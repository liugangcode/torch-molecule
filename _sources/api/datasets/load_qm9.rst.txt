load_qm9
========

QM9 — quantum chemical properties. Deprecated: this function calls ``load_dataset`` and will be removed.

Use instead
-----------

.. code-block:: python

   from torch_molecule.datasets import load_dataset

   data = load_dataset("qm9", targets=["homo", "lumo", "gap"])

Reference
---------

.. autofunction:: torch_molecule.datasets.load_qm9

See :doc:`../../dataset` for caching, custom data, and train/test splits.
