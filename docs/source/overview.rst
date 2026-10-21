Choose a model
==============

Start with the task you want to solve. Models accept molecular SMILES strings
and expose familiar ``fit``, ``predict``, ``encode``, or ``generate`` methods.

.. list-table::
   :header-rows: 1
   :widths: 23 37 40

   * - Your task
     - Start here
     - What you provide
   * - Predict molecular properties
     - :class:`~torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor` or
       :class:`~torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor`
     - SMILES and property labels
   * - Learn molecular representations
     - :class:`~torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder` or
       :class:`~torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder`
     - Unlabeled SMILES
   * - Learn similarity embeddings
     - :class:`~torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder`
     - SMILES, molecular-pair indices, and similarity labels
   * - Generate from molecular demonstrations
     - :class:`~torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator`
     - SMILES, optional demonstration molecules, and normalized scores
   * - Generate with masked sequence diffusion
     - :class:`~torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator`
     - SMILES; an optional scaffold at generation time
   * - Generate toward property values
     - :class:`~torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator`
     - SMILES and property labels
   * - Use a pretrained checkpoint
     - :class:`~torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder` or
       :class:`~torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator`
     - A supported Hugging Face checkpoint identifier

Explore the full model families in the :doc:`api/predictor`,
:doc:`api/encoder`, and :doc:`api/generator` references.

Inputs and outputs
------------------

``X_train`` is normally a list of SMILES strings. Property labels use a numerical
array with one row per molecule and one column per task.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Workflow
     - Training
     - Inference
   * - Property prediction
     - ``fit(X_train, y_train, X_val, y_val)``
     - ``predict(smiles)["prediction"]`` for the GNN predictor
   * - Representation learning
     - ``fit(X_train)`` for self-supervised encoders
     - ``encode(smiles, return_type="np")``
   * - Unconditional generation
     - ``fit(X_train)`` for supporting generators
     - ``generate(n_samples=10)``
   * - SAND similarity learning
     - ``fit(X_train, y_train, pairs=pairs)``
     - ``encode(smiles, return_type="np")``
   * - DemoDiff with demonstrations
     - ``fit(X_train, contexts=indices, score=score)``
     - ``generate(n_samples=10, contexts=smiles, score=score)``

Method-specific inputs are described in each model's API reference. For example,
SAND needs at least two molecular pairs with nonconstant similarity labels.
DemoDiff uses demonstration scores in ``[0, 1]`` and a target score of ``1``;
higher values mean closer to the target. During training its ``contexts`` are
indices into ``X_train``; during generation they are SMILES strings.

Training and pretrained models
------------------------------

Trainable models such as AttrMask, SAND, GenMol, and DemoDiff initialize a new
model through ``fit``. Loading a saved checkpoint is a separate operation.
The Hugging Face pretrained wrappers use ``fit()`` to load their selected
checkpoint; they do not train a new model with that call.

Set architecture and training options in the constructor. Use ``get_params()``
to inspect an estimator's settings and ``set_params()`` to update them before
training. Consult the class reference for its supported settings.

Save and load
-------------

Save a fitted model with ``save_to_local(path)``. Construct the same model class
and call ``load_from_local(path)`` to restore it, then run inference directly.
Do not call ``fit`` again if you want to keep the loaded weights.

.. code-block:: python

   from pathlib import Path
   from torch_molecule import GNNMolecularPredictor

   # predictor is the fitted model from the quickstart.
   output = Path("artifacts/checkpoints")
   output.mkdir(parents=True, exist_ok=True)
   predictor.save_to_local(str(output / "gnn.pt"))

   restored = GNNMolecularPredictor(device="cpu")
   restored.load_from_local(str(output / "gnn.pt"))
   prediction = restored.predict(["CCO"])["prediction"]

See :doc:`example` for the complete training example and :doc:`api/base` for
local and Hugging Face checkpoint methods.
