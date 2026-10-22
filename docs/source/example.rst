Quickstart
==========

Train a predictor, learn molecular embeddings, or generate molecules from
SMILES. These small examples demonstrate the API; use a larger dataset and a
separate evaluation set to assess model quality.

Predict a molecular property
----------------------------

This example learns molecular weight calculated with RDKit. Replace these
values with your own measured property labels for a prediction task.

.. code-block:: python

   import numpy as np
   from rdkit import Chem
   from rdkit.Chem import Descriptors
   from torch_molecule import GNNMolecularPredictor

   smiles = ["CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC", "CCOC", "CCNC"]
   target = np.array([
       [Descriptors.MolWt(Chem.MolFromSmiles(value))] for value in smiles
   ])

   predictor = GNNMolecularPredictor(
       num_task=1,
       task_type="regression",
       num_layer=2,
       hidden_size=32,
       batch_size=4,
       epochs=20,
       device="cpu",
   )
   predictor.fit(
       X_train=smiles[:6],
       y_train=target[:6],
       X_val=smiles[6:],
       y_val=target[6:],
   )
   prediction = predictor.predict(smiles[6:])["prediction"]
   print(prediction.shape)  # (2, 1)

For multiple properties, provide a target array of shape
``(n_molecules, n_tasks)`` and set ``num_task`` accordingly. For automated
hyperparameter search, see ``autofit`` in the :doc:`api/predictor` reference.

Learn molecular embeddings
--------------------------

Attribute masking trains on unlabeled molecules and returns one vector per
molecule. These vectors can be used in similarity analysis or a downstream
predictor.

.. code-block:: python

   from torch_molecule import AttrMaskMolecularEncoder

   smiles = ["CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC", "CCOC", "CCNC"]
   encoder = AttrMaskMolecularEncoder(
       num_layer=2,
       hidden_size=32,
       batch_size=4,
       epochs=20,
       device="cpu",
   )
   encoder.fit(smiles)
   embeddings = encoder.encode(smiles, return_type="np")
   print(embeddings.shape)  # (8, 32)

For similarity-supervised learning, SAND instead takes explicit molecular pairs:
``fit(smiles, similarity_labels, pairs=pair_indices)``. ``pair_indices`` has shape
``(n_pairs, 2)`` and indexes ``smiles``; each pair needs one finite label. At least
two pairs with different labels are required. See :doc:`api/encoder`.

Generate molecules with DemoDiff
--------------------------------

DemoDiff trains on SMILES and can generate with or without demonstrations. This
small configuration trains from scratch using the core dependencies.

.. code-block:: python

   from torch_molecule import DemoDiffMolecularGenerator

   smiles = ["CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC", "CCOC", "CCNC"]
   generator = DemoDiffMolecularGenerator(
       num_layer=2,
       hidden_size=64,
       num_head=4,
       mlp_ratio=2,
       diffusion_steps=20,
       batch_size=4,
       epochs=25,
       learning_rate=1e-3,
       guidance_scale=1.0,
       device="cpu",
   )
   generator.fit(smiles)
   generated = generator.generate(n_samples=5, max_attempts=10)
   print(generated)

Generation returns valid, connected SMILES. If the attempt limit is reached,
DemoDiff warns and returns the available molecules. A short training run can
return fewer molecules than requested.

Use molecular demonstrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For conditional training, pass demonstration indices for every training molecule
and one ``score`` per demonstration. Scores are similarities or normalized task
scores in ``[0, 1]``; the target's score is always ``1``. Use the same score
definition for training and generation.

The repository example computes Morgan fingerprint Tanimoto similarity and
selects similar molecules as demonstrations. Run it from the repository root:

.. code-block:: bash

   python examples/generation/demodiff.py --data data/molecule100.csv --with-context

It trains for 25 epochs, saves and reloads the model, and writes losses and
generated molecules. The script prints the output directory when it finishes.
Use ``--epochs`` or ``--device`` to change the run. See the
:ref:`complete DemoDiff example <DemoDiffMolecularGenerator-example>`
and :doc:`api/generator` for the input format and generation options.

Generate molecules with GenMol
------------------------------

Install the ``genmol`` extra described in :doc:`install`. GenMol trains on
SMILES and supports an optional scaffold at generation time.

.. code-block:: python

   from torch_molecule import GenMolMolecularGenerator

   smiles = ["CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC", "CCOC", "CCNC"]
   generator = GenMolMolecularGenerator(
       num_layer=2,
       num_head=4,
       hidden_size=64,
       intermediate_size=128,
       max_len=64,
       batch_size=4,
       epochs=100,
       learning_rate=1e-3,
       warmup_steps=0,
       ema_decay=0.0,
       device="cpu",
   )
   generator.fit(smiles)
   generated = generator.generate(n_samples=5, min_new_tokens=4)
   print(generated)

GenMol also returns available molecules with a warning when bounded sampling
cannot fill the request. For scaffold generation, pass ``scaffold="c1ccccc1"``
to ``generate`` after training on a suitable molecular dataset.

Use a pretrained generator
--------------------------

Hugging Face wrappers load existing checkpoints. This example needs network
access and the SAFE-GPT dependencies in :doc:`install`.

.. code-block:: python

   from torch_molecule import HFPretrainedMolecularGenerator

   generator = HFPretrainedMolecularGenerator(repo_id="datamol-io/safe-gpt")
   generator.fit()
   generated = generator.generate(n_samples=5)
   print(generated)

More examples
-------------

Browse the maintained scripts for
`prediction <https://github.com/liugangcode/torch-molecule/tree/main/examples/prediction>`_,
`representation learning <https://github.com/liugangcode/torch-molecule/tree/main/examples/encoder>`_,
`generation <https://github.com/liugangcode/torch-molecule/tree/main/examples/generation>`_,
and `datasets <https://github.com/liugangcode/torch-molecule/tree/main/examples/datasets>`_.
See :doc:`dataset` to bring your own data and :doc:`overview` to save a fitted model.
