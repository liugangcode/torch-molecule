# torch-molecule

`torch-molecule` is a developing package for molecular discovery using deep learning, designed with an `sklearn`-style interface for ease of use. The package offers model checkpoints tailored for deployment and benchmarking across various molecular tasks. Current development focuses on three primary components:

1. **Predictive Models**: Includes graph-based models (GREA, SGIR, DCT), RNN-based models (SMILES-RNN, SMILES-Transformers), and others focused on molecular representations.
2. **Generative Models**: Models like Graph DiT, DiGress, GDSS, etc.
3. **Representation Models**: Includes models like MoAMa, AttrMasking, ContextPred, and EdgePred.

> **Note**: This project is actively under development, and features may not yet be stable.

## Project Structure

Here is the current folder structure of `torch_molecule`:

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
   conda create --name torchmolecule python=3.11.7
   ```

2. **Install the package** via pip:

   ```bash
   pip install torch_molecule
   ```

3. For development purposes, install in editable mode:

   ```bash
   pip install -e .
   ```

## Usage

See the `examples` folder for more detailed use cases.

### Python API Example

Below is an example of how to use the `GREAMolecularPredictor` class from `torch_molecule`:

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

# Fit the model with automatic hyperparameter optimization
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

## Acknowledgements

This project is a work in progress, and some features may be unstable.

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template), and we thank the authors for their open-source contribution.