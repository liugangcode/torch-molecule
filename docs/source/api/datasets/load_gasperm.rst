load_gasperm
============

Gas permeability — polymer properties.

.. code-block:: python

   from torch_molecule.datasets import load_gasperm

Reference
---------

.. autofunction:: torch_molecule.datasets.load_gasperm

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_gasperm

   data = load_gasperm(target_cols=["CO2", "N2"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
