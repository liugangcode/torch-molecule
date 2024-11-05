# torch-molecule

`torch-molecule` is an emerging package designed to accelerate molecular discovery using deep learning, featuring an intuitive, `sklearn`-style interface. This package provides model checkpoints for seamless deployment and benchmarking across various molecular tasks. The current development phase focuses on three main components:

1. **Predictive Models**: Includes graph-based models (GREA, SGIR, DCT), RNN-based models (SMILES-RNN, SMILES-Transformers), and other molecular representation models.
2. **Generative Models**: Models such as Graph DiT, DiGress, GDSS, etc.
3. **Representation Models**: Includes MoAMa, AttrMasking, ContextPred, EdgePred, and more.

> **Note**: This project is actively under development; features may be subject to change.

## Project Structure

The current folder structure of `torch_molecule`:

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

2. **Install the package** via pip:

   ```bash
   pip install -i https://test.pypi.org/simple/ torch-molecule
   ```

3. For development, install in editable mode:

   ```bash
   pip install -e .
   ```

## Usage

Refer to the `examples` folder for detailed use cases.

### Python API Example

Here’s how to use the `GREAMolecularPredictor` class from `torch_molecule`:

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

### Using Checkpoints for depolyment

`torch-molecule` provides checkpoints hosted on Hugging Face for common tasks, which can save computational costs by initializing from a pretrained state. For example, you can use a checkpoint for gas permeability predictions (in log10 space):

```python
from torch_molecule import GREAMolecularPredictor

# Use a pre-trained checkpoint from Hugging Face
repo_id = "liuganghuggingface/torch-molecule-ckpt-GREA-gas-separation-logscale"
model = GREAMolecularPredictor()
model.load_model(f"{model_dir}/GREA_{gas}.pt", repo_id=repo_id)
model.set_params(verbose=True)

# Make predictions
predictions = model.predict(smiles_list)
```

### Using Checkpoints for benchmarking

TODO.


## Acknowledgements

This project is under active development, and some features may be subject to change.

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their open-source contribution.
