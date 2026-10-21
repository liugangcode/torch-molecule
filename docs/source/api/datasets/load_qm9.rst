load_qm9
========

QM9 — quantum chemical properties.

.. code-block:: python

   from torch_molecule.datasets import load_qm9

Reference
---------

.. autofunction:: torch_molecule.datasets.load_qm9

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_qm9

   data = load_qm9(target_cols=["homo", "lumo", "gap"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
