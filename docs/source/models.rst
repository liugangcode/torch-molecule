Model catalog
=============

Choose from **31 model classes** for property prediction, molecular representations,
and generation. Models accept SMILES strings and handle their own graph or sequence
representations. Select a model below to see its parameters and method signatures.
Example links download the Python source bundled with this documentation.

Predict properties
------------------

Fit on molecules and labels, then predict regression or classification targets.
Start with GNN for a graph baseline or a SMILES model for a sequence baseline.

.. list-table:: 11 predictors
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Approach
     - Example
   * - :class:`GNN <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor>`
     - GIN or GCN message passing, with optional virtual nodes.
     - :download:`GNN example <../../examples/prediction/gnn.py>`
   * - :class:`GREA <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor>`
     - Graph rationales and environment-based data augmentation.
     - :download:`GREA example <../../examples/prediction/grea.py>`
   * - :class:`SGIR <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor>`
     - Semi-supervised learning for imbalanced regression targets.
     - :download:`SGIR example <../../examples/prediction/sgir.py>`
   * - :class:`GRIN <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor>`
     - Repetition-invariant graph representations for polymers.
     - :download:`GRIN example <../../examples/prediction/grin.py>`
   * - :class:`BFGNN <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor>`
     - Graph learning aligned with the Bellman–Ford algorithm.
     - :download:`BFGNN example <../../examples/prediction/bfgnn.py>`
   * - :class:`DIR <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor>`
     - Learning invariant graph rationales.
     - :download:`DIR example <../../examples/prediction/dir.py>`
   * - :class:`IRM <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor>`
     - Invariant risk minimization with graph neural networks.
     - :download:`IRM example <../../examples/prediction/irm.py>`
   * - :class:`RPGNN <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor>`
     - Relational pooling over node permutations.
     - :download:`RPGNN example <../../examples/prediction/rpgnn.py>`
   * - :class:`SSR <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor>`
     - Regularization for generalization across graph sizes.
     - :download:`SSR example <../../examples/prediction/ssr.py>`
   * - :class:`LSTM <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor>`
     - Recurrent sequence learning from SMILES.
     - :download:`LSTM example <../../examples/prediction/lstm.py>`
   * - :class:`SMILES Transformer <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor>`
     - Transformer sequence learning from SMILES.
     - :download:`Transformer example <../../examples/prediction/smilestransformer.py>`

Learn molecular representations
-------------------------------

Encode molecules as vectors for downstream prediction, clustering, or similarity
analysis. Self-supervised encoders train on unlabeled SMILES; supervised encoders
need the labels or pairs described below.

.. list-table:: 9 encoders
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Training signal
     - Example
   * - :class:`AttrMask <torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder>`
     - Reconstruct masked atom attributes.
     - :download:`AttrMask example <../../examples/encoder/attrmask.py>`
   * - :class:`MoAmA <torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder>`
     - Motif-aware attribute masking and fingerprint reconstruction.
     - :download:`MoAmA example <../../examples/encoder/moama.py>`
   * - :class:`GraphMAE <torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder>`
     - Graph masked autoencoding.
     - :download:`GraphMAE example <../../examples/encoder/graphmae.py>`
   * - :class:`ContextPred <torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder>`
     - Predict molecular subgraph contexts.
     - :download:`ContextPred example <../../examples/encoder/contextpred.py>`
   * - :class:`EdgePred <torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder>`
     - Predict whether pairs of atoms are connected.
     - :download:`EdgePred example <../../examples/encoder/edgepred.py>`
   * - :class:`InfoGraph <torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder>`
     - Learn agreement between graph and substructure representations.
     - :download:`InfoGraph example <../../examples/encoder/infograph.py>`
   * - :class:`Supervised <torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder>`
     - Molecular property labels or computed molecular descriptors.
     - :download:`Supervised example <../../examples/encoder/supervised.py>`
   * - :class:`SAND <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder>`
     - Pairwise similarity scores; dense or compressed molecular descriptors.
     - :doc:`API and input format <api/encoder>`
   * - :class:`Hugging Face <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder>`
     - Load a pretrained sequence encoder, such as ChemBERTa or ChemGPT.
     - :download:`Pretrained encoder example <../../examples/encoder/hfpretrained.py>`

Generate molecules
------------------

Train a generator on molecules or load a pretrained model. Conditioning varies by
method: property labels, molecular scaffolds, and scored demonstrations serve
different purposes and have different input formats.

.. list-table:: 11 generators
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Approach and supported conditioning
     - Example
   * - :class:`DemoDiff <torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator>`
     - Graph diffusion with optional molecular demonstrations and normalized ``score`` values.
     - :download:`DemoDiff example <../../examples/generation/demodiff.py>`
   * - :class:`GenMol <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator>`
     - Masked diffusion over SAFE fragments, optionally retaining a supplied scaffold.
     - :doc:`API and usage <api/generator>`
   * - :class:`GraphDiT <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - Graph diffusion transformer with optional property labels.
     - :download:`GraphDiT example <../../examples/generation/graphdit.py>`
   * - :class:`DeFoG <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - Discrete graph flow matching with optional property labels.
     - :download:`DeFoG example <../../examples/generation/defog.py>`
   * - :class:`DiGress <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator>`
     - Unconditional discrete graph diffusion.
     - :download:`DiGress example <../../examples/generation/digress.py>`
   * - :class:`GDSS <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator>`
     - Unconditional score-based graph diffusion.
     - :download:`GDSS example <../../examples/generation/gdss.py>`
   * - :class:`JT-VAE <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator>`
     - Unconditional junction-tree variational autoencoding.
     - :download:`JT-VAE example <../../examples/generation/jtvae.py>`
   * - :class:`Graph GA <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Genetic search using property labels or a scoring oracle.
     - :download:`Graph GA example <../../examples/generation/graphga.py>`
   * - :class:`MolGPT <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - Autoregressive SMILES transformer with optional properties and scaffold tokens.
     - :download:`MolGPT example <../../examples/generation/molgpt.py>`
   * - :class:`LSTM <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - Recurrent SMILES generation with optional property labels.
     - :download:`LSTM example <../../examples/generation/lstm.py>`
   * - :class:`Hugging Face <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator>`
     - NovoMolGen, MolGen, Molexar, or SAFE-GPT checkpoints; constraints depend on the model family.
     - :download:`SAFE-GPT example <../../examples/generation/pretrained_safe_gpt.py>`

See :doc:`install` for optional dependencies and :doc:`example` for complete
workflows, including dataset preparation and checkpoint loading.
