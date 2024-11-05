# torch-molecule

`torch-molecule` is a package in active development that streamlines molecular discovery through deep learning, with a user-friendly `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. The current scope focuses on three main components:

1. **Predictive Models**: Supports graph-based models (e.g., GREA, SGIR, DCT), RNN-based models (e.g., SMILES-RNN, SMILES-Transformers), and other model based on molecular representation models.
2. **Generative Models**: Models include Graph DiT, DiGress, GDSS, and others.
3. **Representation Models**: Includes MoAMa, AttrMasking, ContextPred, EdgePred, and more.

> **Note**: This project is under active development, and features may change.

## Project Structure

The structure of `torch_molecule`:

```
torch_molecule
├── base.py
├── generator
├── __init__.py
├── predictor
│   ├── components
│   │   ├── gnn_components.py
│   │   └── __init__.py
│   ├── gnn
│   │   ├── architecture.py
│   │   ├── __init__.py
│   │   └── modeling_gnn.py
│   ├── grea
│   │   ├── architecture.py
│   │   ├── __init__.py
│   │   └── modeling_grea.py
│   └── __init__.py
├── representer
└── utils
    ├── format.py
    ├── generic
    │   ├── metrics.py
    │   └── weights.py
    ├── graph
    │   ├── features.py
    │   └── graph_from_smiles.py
    ├── hf_hub.py
    ├── __init__.py
    └── search.py
```

## Installation

1. **Create a Conda environment**:

   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install dependencies**: Dependencies are listed in `requirements.txt`, along with the versions used during development. You can install them by copying and pasting from the `requirements.txt` file and then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install torch_molecule**:

   ```bash
   pip install -i https://test.pypi.org/simple/ torch-molecule
   ```

4. For development mode, install with:

   ```bash
   pip install -e .
   ```

## Usage

See the `examples` folder for more detailed use cases.

### Python API Example

The following example demonstrates how to use the `GREAMolecularPredictor` class from `torch_molecule`:

```python
from torch_molecule import GREAMolecularPredictor

# Initialize the model
model = GREAMolecularPredictor(
    num_tasks=1,
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
    n_trials=10          # Number of trials for hyperparameter optimization
)

# Fit the model with predefined hyperparameters
model = GREAMolecularPredictor(
    num_tasks=1,
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
    X_val=X_val.tolist(),
    y_val=y_val,
)
```

### Using Checkpoints for Deployment

`torch-molecule` provides checkpoints hosted on Hugging Face, which can save computational resources by starting from a pretrained state. For instance, a checkpoint for gas permeability predictions (in log10 space) can be used as follows:

```python
from torch_molecule import GREAMolecularPredictor

# Load a pretrained checkpoint from Hugging Face
repo_id = "liuganghuggingface/torch-molecule-ckpt-GREA-gas-separation-logscale"
model = GREAMolecularPredictor()
model.load_model(f"{model_dir}/GREA_{gas}.pt", repo_id=repo_id)
model.set_params(verbose=True)

# Make predictions
predictions = model.predict(smiles_list)
```

### Using Checkpoints for Benchmarking

_(Coming soon)_

## Acknowledgements

This project is actively developed, and some features may change over time.

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.
