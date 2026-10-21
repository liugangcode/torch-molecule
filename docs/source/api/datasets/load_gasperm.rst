load_gasperm
============

Gas permeability — polymer properties. Deprecated: this function calls ``load_dataset`` and will be removed.

Use instead
-----------

.. code-block:: python

   from torch_molecule.datasets import load_dataset

   data = load_dataset("polymer_gas_permeability", targets=["CO2", "N2"])

Reference
---------

.. autofunction:: torch_molecule.datasets.load_gasperm

See :doc:`../../dataset` for caching, custom data, and train/test splits.
