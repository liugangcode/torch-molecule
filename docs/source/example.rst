Examples
========

This section shows how to use the `torch_molecule` library in practice. More examples are available in the `examples <https://github.com/liugangcode/torch-molecule/tree/main/examples/>`_ folder and the `tests <https://github.com/liugangcode/torch-molecule/tree/main/tests/>`_ folder.


Molecular Property Prediction Usage
-----------------------------------

The following example demonstrates how to use the `GREAMolecularPredictor`:

.. code-block:: python

   from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
   from torch_molecule.utils.search import ParameterType, ParameterSpec

   # Define search parameters
   search_GNN = {
       "gnn_type": ParameterSpec(ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]),
       "norm_layer": ParameterSpec(ParameterType.CATEGORICAL, ["batch_norm", "layer_norm"]),
       "graph_pooling": ParameterSpec(ParameterType.CATEGORICAL, ["mean", "sum", "max"]),
       "augmented_feature": ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan", "maccs", "morgan", None]),
       "num_layer": ParameterSpec(ParameterType.INTEGER, (2, 5)),
       "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 512)),
       "drop_ratio": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
       "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-2)),
       "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-10, 1e-3)),
   }

   search_GREA = {
       "gamma": ParameterSpec(ParameterType.FLOAT, (0.25, 0.75)),
       **search_GNN
   }

   # Train GREA model
   grea_model = GREAMolecularPredictor(
       num_task=num_task,
       task_type="regression",
       model_name="GREA_multitask",
       batch_size=BATCH_SIZE,
       epochs=N_epoch,
       evaluate_criterion='r2',
       evaluate_higher_better=True,
       verbose=True
   )

   # Fit the model
   X_train = ['C1=CC=CC=C1', 'C1=CC=CC=C1']
   y_train = [[0.5], [1.5]]
   X_val = ['C1=CC=CC=C1', 'C1=CC=CC=C1']
   y_val = [[0.5], [1.5]]
   N_trial = 100

   grea_model.autofit(
       X_train=X_train.tolist(),
       y_train=y_train,
       X_val=X_val.tolist(),
       y_val=y_val,
       n_trials=N_trial,
       search_parameters=search_GREA
   )


Molecular Generator Usage
----------------------------

The following example demonstrates how to use the `GraphDITMolecularGenerator` for generating molecules with retry logic for invalid molecules:

.. code-block:: python

   from torch_molecule import GraphDITMolecularGenerator
   from rdkit import Chem

   property_names = ['logP']
   train_smiles_list = ['C1=CC=CC=C1', 'C1=CC=CC=C1']
   train_property_array = [[1], [2]]
   test_property_array = [[1.5], [2.5]]

   # Initialize the generator model
   model_cond = GraphDITMolecularGenerator(
       task_type=['regression'] * len(property_names),
       batch_size=1024,
       drop_condition=0.1,
       verbose=True,
       epochs=10000,
   )

   # Fit the model
   model_cond.fit(train_smiles_list, train_property_array)

   # Generate molecules with retry logic
   max_retries = 10
   generated_smiles_list = model_cond.generate(test_property_array)

   # Function to check if SMILES is valid
   def is_valid_smiles(smiles):
       if smiles is None:
           return False
       mol = Chem.MolFromSmiles(smiles)
       return mol is not None

   # Retry generation for invalid molecules
   for retry in range(max_retries - 1):  # Already did first generation
       # Find indices of invalid SMILES
       invalid_indices = [i for i, smiles in enumerate(generated_smiles_list) if not is_valid_smiles(smiles)]
       
       if not invalid_indices:
           print(f"All SMILES valid after {retry + 1} attempts")
           break
           
       print(f"Retry {retry + 1}: Regenerating {len(invalid_indices)} invalid molecules")
       
       # Extract properties for invalid molecules
       invalid_properties = test_property_array[invalid_indices]
       
       # Regenerate only for invalid molecules
       new_smiles = model_cond.generate(invalid_properties)
       
       # Replace invalid molecules with new generations
       for idx, new_idx in enumerate(invalid_indices):
           generated_smiles_list[new_idx] = new_smiles[idx]
       
       if retry == max_retries - 2:  # Last iteration
           print(f"Reached maximum retries ({max_retries}). {len(invalid_indices)} molecules still invalid.")


Using Pretrained Checkpoints
----------------------------

`torch_molecule` supports loading and saving models via Hugging Face Hub.

.. code-block:: python

   from torch_molecule import GREAMolecularPredictor
   from sklearn.metrics import mean_absolute_error

   # huggingface repo_id including the user name and repo name
   repo_id = "user/repo_id"

   # Train and push a model to Hugging Face
   model = GREAMolecularPredictor()
   model.autofit(
       X_train=X.tolist(),
       y_train=y_train,
       X_val=X_val.tolist(),
       y_val=y_val,
       n_trials=100
   )

   output = model.predict(X_test.tolist())
   mae = mean_absolute_error(y_test, output['prediction'])
   metrics = {'MAE': mae}

   model.push_to_huggingface(
       repo_id=repo_id,
       task_id=f"{task_name}",
       metrics=metrics,
       commit_message=f"Upload GREA_{task_name} model with metrics: {metrics}",
       private=False
   )

   # Load a pretrained model checkpoint
   model_dir = "local_model_dir_to_save"
   model = GREAMolecularPredictor()
   model.load_model(f"{model_dir}/GREA_{task_name}.pt", repo_id=repo_id)
   model.set_params(verbose=True)

   predictions = model.predict(smiles_list)
