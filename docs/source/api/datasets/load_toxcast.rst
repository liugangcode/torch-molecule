load_toxcast
============

ToxCast — toxicity assays.

.. code-block:: python

   from torch_molecule.datasets import load_toxcast

Reference
---------

.. autofunction:: torch_molecule.datasets.load_toxcast

Example
-------

.. code-block:: python

   from torch_molecule.datasets import load_toxcast

   data = load_toxcast(target_cols=["ACEA_T47D_80hr_Negative"])
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, n_selected_targets)

See :doc:`../../dataset` for caching, custom data, and train/test splits.
