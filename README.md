# torch-molecule

`torch-molecule` is a package under active development that facilitates molecular discovery through deep learning, featuring a user-friendly, `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. Currently, the package focuses on three main components:

1. **Predictive Models**: Supports graph-based models (e.g., GREA, SGIR, DCT), RNN-based models (e.g., SMILES-RNN, SMILES-Transformers), and other models based on molecular representations.
2. **Generative Models**: Includes models such as Graph DiT, DiGress, GDSS, and others.
3. **Representation Models**: Includes MoAMa, AttrMasking, ContextPred, EdgePred, and more.

> **Note**: This project is in active development, and features may change.

## Project Structure

The structure of `torch_molecule` is as follows:

`tree -L 2 torch_molecule -I '__pycache__|*.pyc|*.pyo|.git|old*'`

```
torch_molecule
├── base
│   ├── base.py
│   ├── encoder.py
│   ├── generator.py
│   ├── __init__.py
│   └── predictor.py
├── encoder
│   └── supervised
├── generator
├── __init__.py
├── nn
│   ├── gnn.py
│   ├── __init__.py
│   └── mlp.py
├── predictor
│   ├── gnn
│   ├── grea
│   ├── __init__.py
│   └── sgir
└── utils
    ├── checker.py
    ├── checkpoint.py
    ├── format.py
    ├── generic
    ├── graph
    ├── hf.py
    ├── __init__.py
    └── search.py
```

## Installation

1. **Create a Conda environment**:

   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install dependencies**: Dependencies are listed in `requirements.txt`, with versions used during development. Install them by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install torch_molecule**:

   ```bash
   pip install -i https://test.pypi.org/simple/ torch-molecule
   ```

## Usage

Refer to the `examples` folder for detailed use cases.

### Python API Example

The following example demonstrates how to use the `GREAMolecularPredictor` class from `torch_molecule`:

```python
from torch_molecule import GREAMolecularPredictor

# Initialize the model
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

# Fit the model with hyperparameter optimization
model.autofit(
    X_train=X.tolist(),  # List of SMILES strings
    y_train=y_train,     # numpy array [n_samples, n_tasks]
    X_val=X_val.tolist(),
    y_val=y_val,
    n_trials=100          # Number of trials for hyperparameter optimization
)

# Fit the model with predefined hyperparameters
model = GREAMolecularPredictor(
    num_task=1,
    task_type="regression",
    num_layer=5,         # Specify hyperparameter
    model_name=f"GREA_{task_name}",
    batch_size=512,
    epochs=500,
    evaluate_criterion='r2',
    evaluate_higher_better=True,
    verbose=True
)

model.fit(
    X_train=X.tolist(),  # List of SMILES strings
    y_train=y_train,     # numpy array [n_samples, n_tasks]
    X_val=None, # leave it None if the same as the train
    y_val=None,
)
```

### Using Checkpoints for Deployment

`torch-molecule` provides checkpoints hosted on Hugging Face, which can save computational resources by starting from a pretrained state. For example, a checkpoint for gas permeability predictions (in log10 space) can be used as follows:

```python
from torch_molecule import GREAMolecularPredictor

repo_id = "user/repo_id"
# Push a trained model to Hugging Face
model = GREAMolecularPredictor()
model.autofit(
    X_train=X.tolist(),  # List of SMILES strings
    y_train=y_train,     # numpy array [n_samples, n_tasks]
    X_val=X_val.tolist(),
    y_val=y_val,
    n_trials=100          # Number of trials for hyperparameter optimization
)
output = model.predict(X_test.tolist()) # (n_sample, n_task)
mae = mean_absolute_error(y_test, output['prediction'])
metrics = {'MAE': mae}
model.push_to_huggingface(
    repo_id=repo_id,
    task_id=f"{task_name}",
    metrics=metrics,
    commit_message=f"Upload GREA_{task_name} model with metrics: {metrics}",
    private=False
)
# Load a pretrained checkpoint from Hugging Face
model = GREAMolecularPredictor()
model.load_model(f"{model_dir}/GREA_{task_name}.pt", repo_id=repo_id)
model.set_params(verbose=True)

# Make predictions
predictions = model.predict(smiles_list)
```

### Using Checkpoints for Benchmarking

_(Coming soon)_

## Acknowledgements

This project is under active development, and some features may change over time.

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.
