Generator API
=============

Fit a model on SMILES with ``fit(...)`` and sample molecules with
``generate(...)``. Conditional inputs and sample-count arguments differ by
model; use the signatures below when switching methods.

.. list-table:: Generation and conditioning
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Optional conditioning
   * - :class:`DemoDiff <torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator>`
     - Demonstration SMILES in ``contexts`` and normalized ``score`` values.
   * - :class:`GenMol <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator>`
     - A molecular substructure in ``scaffold``.
   * - :class:`GraphDiT <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`DeFoG <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`Graph GA <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Property targets in ``labels`` or a scoring oracle configured on the estimator.
   * - :class:`MolGPT <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - ``properties`` and ``scaffolds``.
   * - :class:`LSTM <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`Hugging Face <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator>`
     - Scaffold, fragment, or property inputs, depending on the checkpoint family.
   * - :class:`DiGress <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator>`
     - Unconditional generation.
   * - :class:`GDSS <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator>`
     - Unconditional generation.
   * - :class:`JT-VAE <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator>`
     - Unconditional generation.

Use ``save_to_local(path)`` and ``load_from_local(path)`` to save and restore
trained models. Additional persistence methods are documented in :doc:`base`;
Graph GA does not support Hugging Face uploads or downloads.

See the :doc:`model catalog <../models>` for method summaries and links to
examples. Optional dependencies are listed in :doc:`../install`.

Graph models
------------

.. rubric:: DemoDiff: generation with demonstrations

DemoDiff trains on SMILES and optionally uses other molecules as demonstrations.
In ``fit()``, ``contexts[i]`` lists indices of other training molecules for target
``i``; in ``generate()``, ``contexts`` contains demonstration SMILES. Supply
``score`` as normalized values in [0, 1], with higher values meaning more
target-like demonstrations. The target's score is fixed to 1 internally.
Use the same score definition in training and generation. The local example
uses Morgan fingerprint Tanimoto similarity: 1 means identical fingerprints
and 0 means no shared fingerprint bits. During generation, scores describe the
demonstrations in the desired task; they do not guarantee measured similarities
of generated molecules. Omit both ``contexts`` and ``score`` for
unconditional training or generation. No additional packages are required.

.. code-block:: python

   from torch_molecule import DemoDiffMolecularGenerator

   smiles = ["CCOCCN", "c1ccccc1CCO", "CC(=O)NCCO"]
   model = DemoDiffMolecularGenerator(
       num_layer=2, hidden_size=64, num_head=4,
       diffusion_steps=50, epochs=20, batch_size=3,
   )
   model.fit(
       smiles,
       contexts=[[1], [0], [0, 1]],
       score=[[0.8], [0.8], [0.6, 0.7]],
   )
   generated = model.generate(
       n_samples=10, contexts=[smiles[0]], score=[0.8],
   )

The scores above are illustrative. Demonstrations guide generation;
they are not guaranteed substructures. The tokenizer can lose stereochemistry
and formal charges, while polymer attachment atoms are supported.
Retrain checkpoints previously trained with distance values and a target score of 0.

.. autoclass:: torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: GraphDiT: graph diffusion transformers
.. autoclass:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: DeFoG: discrete flow matching
.. autoclass:: torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph GA: genetic search
.. autoclass:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, save_to_hf, load_from_hf
   :members: fit, generate
   :undoc-members:
   :show-inheritance:
   
.. automodule:: torch_molecule.generator.graph_ga.oracle
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: DiGress: discrete graph diffusion
.. autoclass:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_class, dataset_info, model_name
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: GDSS: score-based graph diffusion
.. autoclass:: torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: JT-VAE: junction-tree variational autoencoding
.. autoclass:: torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

Sequence models
---------------

.. rubric:: GenMol: masked diffusion over SAFE fragments

GenMol trains on SMILES strings and generates molecules using masked diffusion
over SAFE fragments. Pass ``scaffold="c1ccccc1"`` to ``generate()`` to retain a
benzene substructure in generated molecules. Install its optional packages with
``pip install ".[genmol]"`` from the source checkout.

.. autoclass:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate, load_from_pretrained
   :undoc-members:
   :show-inheritance:

.. rubric:: MolGPT: autoregressive SMILES generation
.. autoclass:: torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: LSTM: recurrent SMILES generation
.. autoclass:: torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

Pretrained generators
---------------------

Hugging Face checkpoints can be loaded for de novo and constrained generation
via :class:`~torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator`.
Supported families include NovoMolGen, MolGen, Molexar, and SAFE-GPT. Extra
packages may be required; see :doc:`../install`.

.. rubric:: Hugging Face Pretrained Generators
.. autoclass:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate, save_to_local, load_from_local
   :undoc-members:
   :show-inheritance:
