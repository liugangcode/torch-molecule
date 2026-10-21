Datasets
========

Model inputs are lists of SMILES strings. For supervised prediction, add a
numerical target array with shape ``(n_molecules, n_tasks)``. Dataset loaders
return a :class:`~torch_molecule.datasets.constant.SMILESDataset` with ``data``
and ``target`` attributes; unlabeled datasets have ``target=None``.

Load a dataset
--------------

QM9 and the other Hub datasets download on first use and reuse files in your
chosen directory. Select properties with ``target_cols``.

.. code-block:: python

   from torch_molecule.datasets import load_qm9

   data = load_qm9(
       local_dir="artifacts/datasets/qm9",
       target_cols=["homo", "lumo", "gap"],
   )
   print(len(data.data))
   print(data.target.shape)  # (n_molecules, 3)

The polymer gas-permeability dataset is included with the package:

.. code-block:: python

   from torch_molecule.datasets import load_gasperm

   data = load_gasperm(target_cols=["CO2", "N2"])
   print(data.target.shape)  # (n_polymers, 2)

Bring your own data
-------------------

Pass SMILES directly to a model, or wrap them in ``SMILESDataset`` to use the
built-in splitting tools. For an unlabeled CSV with a ``smiles`` column:

.. code-block:: python

   import csv
   from torch_molecule.datasets import SMILESDataset

   with open("data/molecule100.csv", newline="") as handle:
       smiles = [row["smiles"] for row in csv.DictReader(handle)]

   data = SMILESDataset(data=smiles, target=None)

The source checkout includes ``data/molecule100.csv`` and ``data/polymer100.csv``
for small local examples. Both contain SMILES; the polymer dataset uses ``*``
attachment atoms. Check the selected model's support for polymer inputs and
attachment atoms before using it for polymer generation.

Choose a split
--------------

.. code-block:: python

   from torch_molecule.datasets import load_gasperm

   data = load_gasperm(target_cols=["CO2"])
   train, validation = data.train_test_split(
       test_size=0.2,
       method="random",
       seed=42,
   )

Pass ``train.data`` and ``train.target`` to ``fit``; supply the validation data
through ``X_val`` and ``y_val`` when the model supports them. Keep a separate
test set for a final performance estimate.

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Method
     - Holdout strategy
     - Useful for
   * - ``random``
     - Randomly assign molecules to each split
     - An initial baseline
   * - ``scaffold``
     - Keep Bemis–Murcko scaffold groups separate
     - Testing on unseen molecular scaffolds
   * - ``butina``
     - Separate clusters based on Morgan fingerprint Tanimoto similarity
     - Testing on different structural neighborhoods
   * - ``size``
     - Split by heavy-atom count
     - Testing transfer across molecular sizes

Use ``subsample(n=100, seed=42)`` for a quick local experiment. Evaluate
benchmarks on the intended dataset and split rather than a debugging subset.

Loader reference
----------------

.. rubric:: QM9 — quantum chemical properties

.. autofunction:: torch_molecule.datasets.load_qm9

.. rubric:: ChEMBL2k — bioactive molecules

.. autofunction:: torch_molecule.datasets.load_chembl2k

.. rubric:: Broad6k — bioactive molecules

.. autofunction:: torch_molecule.datasets.load_broad6k

.. rubric:: ToxCast — toxicity assays

.. autofunction:: torch_molecule.datasets.load_toxcast

.. rubric:: ADMET — pharmacokinetic and toxicity properties

.. autofunction:: torch_molecule.datasets.load_admet

.. rubric:: Gas permeability — polymer properties

.. autofunction:: torch_molecule.datasets.load_gasperm

.. rubric:: ZINC250k — unlabeled molecules

.. autofunction:: torch_molecule.datasets.load_zinc250k

Dataset interface
-----------------

.. autoclass:: torch_molecule.datasets.constant.SMILESDataset
   :members: subsample, train_test_split
