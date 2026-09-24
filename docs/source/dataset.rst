Datasets Overview
====================================================

The ``torch-molecule`` package provides easy access to a wide range of datasets
(locally or from the Hugging Face Hub). Loaders return a
:class:`~torch_molecule.datasets.constant.SMILESDataset` with ``.data``
(SMILES strings) and ``.target`` (property array, or ``None`` for unlabeled sets).

Built-in loaders
----------------

.. rubric:: QM9
.. autofunction:: torch_molecule.datasets.load_qm9

.. rubric:: ChEMBL2k (a small subset from ChEMBL database)
.. autofunction:: torch_molecule.datasets.load_chembl2k

.. rubric:: Broad6k
.. autofunction:: torch_molecule.datasets.load_broad6k

.. rubric:: ToxCast (The US EPA Toxicity Forecaster program)
.. autofunction:: torch_molecule.datasets.load_toxcast

.. rubric:: ADMET (absorption-distribution-metabolism-excretion-toxicity)
.. autofunction:: torch_molecule.datasets.load_admet

.. rubric:: Gas Permeability for polymers
.. autofunction:: torch_molecule.datasets.load_gasperm

.. rubric:: ZINC250k (unlabeled; generation / screening)
.. autofunction:: torch_molecule.datasets.load_zinc250k

``SMILESDataset`` and splitting
-------------------------------

.. autoclass:: torch_molecule.datasets.constant.SMILESDataset
   :members: subsample, train_test_split
   :undoc-members:
   :show-inheritance:

Supported ``train_test_split`` methods:

- ``"random"``: i.i.d. baseline split
- ``"scaffold"``: hold out unseen Bemis-Murcko scaffolds
- ``"butina"``: hold out unseen Taylor-Butina clusters (Morgan / Tanimoto)
- ``"size"``: split by heavy-atom count

Example:

.. code-block:: python

   from torch_molecule.datasets import load_qm9

   data = load_qm9(local_dir="torchmol_data")
   # Split the full dataset. subsample() is only for local debugging / CI —
   # do not shrink a benchmark just to make Butina cheaper.
   train, val = data.train_test_split(test_size=0.2, method="scaffold", seed=42)

   smiles_list, property_np_array = train.data, train.target
