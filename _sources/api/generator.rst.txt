Molecular Generation Models
=============================

The generator models inherit from the :class:`torch_molecule.base.generator.BaseMolecularGenerator` class and share common methods for model training, generation and persistence.

The following models support conditional generation (click model name to jump to details):

.. list-table:: Conditional Generation Models
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - :class:`GraphDITMolecularGenerator <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - `Graph Diffusion Transformers for Multi-Conditional Molecular Generation <https://arxiv.org/abs/2401.13858>`_
   * - :class:`DeFoGMolecularGenerator <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - `Discrete Flow Matching for Graph Generation <https://openreview.net/forum?id=KPRIwWhqAZ>`_
   * - :class:`GraphGAMolecularGenerator <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Graph Genetic Algorithm with Random Forests
   * - :class:`MolGPTMolecularGenerator <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - `MolGPT: Molecular Generation Using a Transformer-Decoder Model <https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600>`_
   * - :class:`LSTMMolecularGenerator <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - LSTM

.. rubric:: Training and Generation

- ``fit(X, **kwargs)``: Train the model on given data, where X contains SMILES strings (y should be provided for conditional generation)
- ``generate(n_samples, **kwargs)``: Generate new molecules and return a list of SMILES strings (y should be provided for conditional generation)

.. rubric:: Model Persistence

inherited from :class:`torch_molecule.base.base.BaseModel`

- ``save_to_local(path)``: Save the trained model to a local file
- ``load_from_local(path)``: Load a trained model from a local file
- ``save_to_hf(repo_id)``: Push the model to Hugging Face Hub

  Not implemented for:
  - :class:`torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator`

- ``load_from_hf(repo_id, local_cache)``: Load a model from Hugging Face Hub and save it to a local file

  Not implemented for:
  - :class:`torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator`

- ``save(path, repo_id)``: Save the model to either local storage or Hugging Face
- ``load(path, repo_id)``: Load a model from either local storage or Hugging Face

Modeling Molecules as Graphs
---------------------------------------------------------------------

.. rubric:: GraphDiT for Unconditional/Multi-Conditional Molecular Generation
.. autoclass:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: Discrete Flow Matching for Graph Generation for Unconditional/Multi-Conditional Molecular Generation
.. autoclass:: torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph Genetic Algorithm for Unconditional/Multi-Conditional Molecular Generation
.. autoclass:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, save_to_hf, load_from_hf
   :members: fit, generate
   :undoc-members:
   :show-inheritance:
   
.. automodule:: torch_molecule.generator.graph_ga.oracle
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: DiGress for Unconditional Molecular Generation
.. autoclass:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_class, dataset_info, model_name
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: GDSS for Unconditional Molecular Generation
.. autoclass:: torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: JT-VAE for Unconditional Molecular Generation
.. autoclass:: torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

Modeling Molecules as Sequences
--------------------------------

.. rubric:: MolGPT for Unconditional/Multi-Conditional Molecular Generation
.. autoclass:: torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: LSTM for Unconditional/Multi-Conditional Molecular Generation
.. autoclass:: torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance: