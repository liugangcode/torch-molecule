<p align="center">
  <img src="assets/logo.png" alt="torch-molecule logo" width="600"/>
</p>

<p align="center">
  <a href="https://github.com/liugangcode/torch-molecule">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub Repository">
  </a>
  <a href="https://liugangcode.github.io/torch-molecule/">
    <img src="https://img.shields.io/badge/Documentation-Online-brightgreen?logo=readthedocs" alt="Documentation">
  </a>
</p>

<p align="center">
  <b>Deep learning for molecular discovery with a simple sklearn-style interface</b>
</p>

---

`torch-molecule` is a package under active development that facilitates molecular discovery through deep learning, featuring a user-friendly, `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. Currently, the package focuses on three main components:

1. **Predictive Models**: Done: GREA, SGIR, IRM, GIN/GCN w/ virtual, DIR. TODO: SMILES-based LSTM/Transformers, more
2. **Generative Models**: Done: Graph DiT, GraphGA, DiGress. TODO:, GDSS, more
3. **Representation Models**: Done: MoAMa, AttrMasking, ContextPred, EdgePred. TODO: checkpoints, more 

> **Note**: This project is in active development, and features may change.

## Installation

1. **Create a Conda environment**:
   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install `torch_molecule` from GitHub**:

   Clone the repository:
   ```bash
   git clone https://github.com/liugangcode/torch-molecule
   ```

   Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

   Editable install:
   ```bash
   pip install -e .
   ```

3. **Install `torch_molecule` from PyPI** (Legacy):
   ```bash
   pip install -i https://test.pypi.org/simple/ torch-molecule
   ```
## Usage

Refer to the `tests` folder for more use cases.

### Python API Example

The following example demonstrates how to use the `GREAMolecularPredictor` class from `torch_molecule`:

More examples could be found in the folders `examples` and `tests`.

```python
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

<!-- ### Using Checkpoints for Benchmarking

_(Coming soon)_ -->


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
│   ├── attrmask
│   ├── constant.py
│   ├── contextpred
│   ├── edgepred
│   ├── moama
│   └── supervised
├── generator
│   ├── digress
│   ├── graph_dit
│   └── graphga
├── __init__.py
├── nn
│   ├── attention.py
│   ├── embedder.py
│   ├── gnn.py
│   ├── __init__.py
│   └── mlp.py
├── predictor
│   ├── dir
│   ├── gnn
│   ├── grea
│   ├── irm
│   ├── lstm
│   ├── rpgnn
│   ├── sgir
│   └── ssr
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

## Acknowledgements

This project is under active development, and some features may change over time.

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.