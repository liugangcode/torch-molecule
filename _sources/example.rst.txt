Examples
========

This section shows how to use the `torch_molecule` library in practice. More examples are available in the `tests/` folder.

Basic Usage
-----------

The following example demonstrates how to use the `GREAMolecularPredictor`:

.. code-block:: python

   from torch_molecule import GREAMolecularPredictor

   model = GREAMolecularPredictor(
       num_task=1,
       task_type="regression",
       model_name=f"GREA_{task_name}",
       batch_size=512,
       epochs=500,
       evaluate_criterion='r2',
       evaluate_higher_better=True,
       verbose=True
   )

   # Fit with hyperparameter optimization
   model.autofit(
       X_train=X.tolist(),
       y_train=y_train,
       X_val=X_val.tolist(),
       y_val=y_val,
       n_trials=100
   )

   # Or fit with manually specified hyperparameters
   model = GREAMolecularPredictor(
       num_task=1,
       task_type="regression",
       num_layer=5,
       model_name=f"GREA_{task_name}",
       batch_size=512,
       epochs=500,
       evaluate_criterion='r2',
       evaluate_higher_better=True,
       verbose=True
   )

   model.fit(
       X_train=X.tolist(),
       y_train=y_train,
       X_val=None,
       y_val=None,
   )

Using Pretrained Checkpoints
----------------------------

`torch_molecule` supports loading and saving models via Hugging Face Hub.

.. code-block:: python

   from torch_molecule import GREAMolecularPredictor
   from sklearn.metrics import mean_absolute_error

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
   model = GREAMolecularPredictor()
   model.load_model(f"{model_dir}/GREA_{task_name}.pt", repo_id=repo_id)
   model.set_params(verbose=True)

   predictions = model.predict(smiles_list)

