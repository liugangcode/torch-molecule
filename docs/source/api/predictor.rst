Predictor API
=============

Train molecular property models with ``fit(X_train, y_train)`` and predict on
new SMILES with ``predict(X)``. Predictions are returned in a dictionary; see
each class for its output fields and task-specific options.

.. list-table:: Common operations
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``fit(...)``
     - Train with SMILES and regression or classification labels.
   * - ``autofit(...)``
     - Search hyperparameters with Optuna and train a model.
   * - ``predict(X)``
     - Predict properties for a list of SMILES.
   * - ``save_to_local(path)`` / ``load_from_local(path)``
     - Save or restore a trained model.

Compare all 11 predictors in the :doc:`model catalog <../models>`. Checkpoint
and Hugging Face methods are documented in :doc:`base`.


Model reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Class
     - Purpose
   * - :class:`GNNMolecularPredictor <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor>`
     - GIN or GCN message passing, with optional virtual nodes.

   * - :class:`GREAMolecularPredictor <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor>`
     - Graph rationales and environment-based data augmentation.

   * - :class:`SGIRMolecularPredictor <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor>`
     - Semi-supervised learning for imbalanced regression targets.

   * - :class:`GRINMolecularPredictor <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor>`
     - Repetition-invariant graph representations for polymers.

   * - :class:`BFGNNMolecularPredictor <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor>`
     - Graph learning aligned with the Bellman–Ford algorithm.

   * - :class:`DIRMolecularPredictor <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor>`
     - Learning invariant graph rationales.

   * - :class:`IRMMolecularPredictor <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor>`
     - Invariant risk minimization with graph neural networks.

   * - :class:`RPGNNMolecularPredictor <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor>`
     - Relational pooling over node permutations.

   * - :class:`SSRMolecularPredictor <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor>`
     - Regularization for generalization across graph sizes.

   * - :class:`LSTMMolecularPredictor <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor>`
     - Recurrent sequence learning from SMILES.

   * - :class:`SMILESTransformerMolecularPredictor <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor>`
     - Transformer sequence learning from SMILES.

.. toctree::
   :hidden:

   GNN <models/GNNMolecularPredictor>
   GREA <models/GREAMolecularPredictor>
   SGIR <models/SGIRMolecularPredictor>
   GRIN <models/GRINMolecularPredictor>
   BFGNN <models/BFGNNMolecularPredictor>
   DIR <models/DIRMolecularPredictor>
   IRM <models/IRMMolecularPredictor>
   RPGNN <models/RPGNNMolecularPredictor>
   SSR <models/SSRMolecularPredictor>
   LSTM <models/LSTMMolecularPredictor>
   SMILES Transformer <models/SMILESTransformerMolecularPredictor>

.. raw:: html

   <a class="tm-api-redirect" id="torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor" href="models/GNNMolecularPredictor.html#torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor">GNNMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.fit" href="models/GNNMolecularPredictor.html#torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.fit">GNNMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.autofit" href="models/GNNMolecularPredictor.html#torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.autofit">GNNMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.predict" href="models/GNNMolecularPredictor.html#torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.predict">GNNMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor" href="models/GREAMolecularPredictor.html#torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor">GREAMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.fit" href="models/GREAMolecularPredictor.html#torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.fit">GREAMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.autofit" href="models/GREAMolecularPredictor.html#torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.autofit">GREAMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.predict" href="models/GREAMolecularPredictor.html#torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.predict">GREAMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor" href="models/SGIRMolecularPredictor.html#torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor">SGIRMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.fit" href="models/SGIRMolecularPredictor.html#torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.fit">SGIRMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.autofit" href="models/SGIRMolecularPredictor.html#torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.autofit">SGIRMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.predict" href="models/SGIRMolecularPredictor.html#torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.predict">SGIRMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor" href="models/GRINMolecularPredictor.html#torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor">GRINMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.fit" href="models/GRINMolecularPredictor.html#torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.fit">GRINMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.autofit" href="models/GRINMolecularPredictor.html#torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.autofit">GRINMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.predict" href="models/GRINMolecularPredictor.html#torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.predict">GRINMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor" href="models/BFGNNMolecularPredictor.html#torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor">BFGNNMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.fit" href="models/BFGNNMolecularPredictor.html#torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.fit">BFGNNMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.autofit" href="models/BFGNNMolecularPredictor.html#torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.autofit">BFGNNMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.predict" href="models/BFGNNMolecularPredictor.html#torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.predict">BFGNNMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor" href="models/DIRMolecularPredictor.html#torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor">DIRMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.fit" href="models/DIRMolecularPredictor.html#torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.fit">DIRMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.autofit" href="models/DIRMolecularPredictor.html#torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.autofit">DIRMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.predict" href="models/DIRMolecularPredictor.html#torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.predict">DIRMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor" href="models/IRMMolecularPredictor.html#torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor">IRMMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.fit" href="models/IRMMolecularPredictor.html#torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.fit">IRMMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.autofit" href="models/IRMMolecularPredictor.html#torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.autofit">IRMMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.predict" href="models/IRMMolecularPredictor.html#torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.predict">IRMMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor" href="models/RPGNNMolecularPredictor.html#torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor">RPGNNMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.fit" href="models/RPGNNMolecularPredictor.html#torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.fit">RPGNNMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.autofit" href="models/RPGNNMolecularPredictor.html#torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.autofit">RPGNNMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.predict" href="models/RPGNNMolecularPredictor.html#torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.predict">RPGNNMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor" href="models/LSTMMolecularPredictor.html#torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor">LSTMMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.fit" href="models/LSTMMolecularPredictor.html#torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.fit">LSTMMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.autofit" href="models/LSTMMolecularPredictor.html#torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.autofit">LSTMMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.predict" href="models/LSTMMolecularPredictor.html#torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.predict">LSTMMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor" href="models/SMILESTransformerMolecularPredictor.html#torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor">SMILESTransformerMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.fit" href="models/SMILESTransformerMolecularPredictor.html#torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.fit">SMILESTransformerMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.autofit" href="models/SMILESTransformerMolecularPredictor.html#torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.autofit">SMILESTransformerMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.predict" href="models/SMILESTransformerMolecularPredictor.html#torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.predict">SMILESTransformerMolecularPredictor.predict reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor" href="models/SSRMolecularPredictor.html#torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor">SSRMolecularPredictor reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.fit" href="models/SSRMolecularPredictor.html#torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.fit">SSRMolecularPredictor.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.autofit" href="models/SSRMolecularPredictor.html#torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.autofit">SSRMolecularPredictor.autofit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.predict" href="models/SSRMolecularPredictor.html#torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.predict">SSRMolecularPredictor.predict reference</a>
