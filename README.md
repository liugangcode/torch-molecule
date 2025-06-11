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

`torch-molecule` is a package under active development that facilitates molecular discovery through deep learning, featuring a user-friendly, `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. Currently, the package focuses on three main components: **Predictive Models**, **Generative Models**, and **Representation Models**. See the [List of Supported Models](#list-of-supported-models) section for all available models.

## Installation

1. **Create a Conda environment**:
   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install using pip (0.1.0)**:

   ```bash
   pip install torch-molecule
   ```

3. **Install from source for the latest version**:

   Clone the repository:

   ```bash
   git clone https://github.com/liugangcode/torch-molecule
   cd torch-molecule
   ```

   Install:

   ```bash
   pip install .
   ```

### Additional Packages

| Model | Required Packages |
|-------|-------------------|
| HFPretrainedMolecularEncoder | transformers |
| BFGNNMolecularPredictor | torch-scatter |
| GRINMolecularPredictor | torch-scatter |

**For models that require `torch-scatter`: Install using the following command: `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`, e.g.,

> `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html`

**For models that require `transformers`:** `pip install transformers`

## Usage

Refer to the `tests` folder for more use cases.

### Python API Example

The following example demonstrates how to use the `GREAMolecularPredictor` class from `torch_molecule`:

More examples could be found in the folders `examples` and `tests`.

```python
from torch_molecule import GREAMolecularPredictor

# Train GREA model
grea_model = GREAMolecularPredictor(
    num_task=num_task,
    task_type="regression",
    model_name="GREA_multitask",
    evaluate_criterion='r2',
    evaluate_higher_better=True,
    verbose=True
)

# Fit the model
X_train = ['C1=CC=CC=C1', 'C1=CC=CC=C1']
y_train = [[0.5], [1.5]]
X_val = ['C1=CC=CC=C1', 'C1=CC=CC=C1']
y_val = [[0.5], [1.5]]
N_trial = 10

grea_model.autofit(
    X_train=X_train.tolist(),
    y_train=y_train,
    X_val=X_val.tolist(),
    y_val=y_val,
    n_trials=N_trial,
)
```

### Checkpoints

`torch-molecule` provides checkpoint functions that can be interacted with on Hugging Face.

```python
from torch_molecule import GREAMolecularPredictor
from sklearn.metrics import mean_absolute_error

# Define the repository ID for Hugging Face
repo_id = "user/repo_id"

# Initialize the GREAMolecularPredictor model
model = GREAMolecularPredictor()

# Train the model using autofit
model.autofit(
    X_train=X.tolist(),  # List of SMILES strings for training
    y_train=y_train,     # numpy array [n_samples, n_tasks] for training labels
    X_val=X_val.tolist(),# List of SMILES strings for validation
    y_val=y_val,         # numpy array [n_samples, n_tasks] for validation labels
)

# Make predictions on the test set
output = model.predict(X_test.tolist()) # (n_sample, n_task)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, output['prediction'])
metrics = {'MAE': mae}

# Save the trained model to Hugging Face
model.save_to_hf(
    repo_id=repo_id,
    task_id=f"{task_name}",
    metrics=metrics,
    commit_message=f"Upload GREA_{task_name} model with metrics: {metrics}",
    private=False
)

# Load a pretrained checkpoint from Hugging Face
model = GREAMolecularPredictor()
model.load_from_hf(repo_id=repo_id, local_cache=f"{model_dir}/GREA_{task_name}.pt")

# Set model parameters
model.set_params(verbose=True)

# Make predictions using the loaded model
predictions = model.predict(smiles_list)
```

## List of Supported Models

### Predictive Models

| Model                | Reference           |
|----------------------|---------------------|
| GRIN                 | [Learning Repetition-Invariant Representations for Polymer Informatics. May 2025](https://arxiv.org/abs/2505.10726) |
| BFGNN                 | [Graph neural networks extrapolate out-of-distribution for shortest paths. March 2025](https://arxiv.org/abs/2503.19173) |
| SGIR                 | [Semi-Supervised Graph Imbalanced Regression. KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599497) |
| GREA                | [Graph Rationalization with Environment-based Augmentations. KDD 2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539347) |
| DIR                  | [Discovering Invariant Rationales for Graph Neural Networks. ICLR 2022](https://arxiv.org/abs/2201.12872) |
| SSR                  | [SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks. NeurIPS 2022](https://arxiv.org/abs/2206.07096) |
| IRM                  | [Invariant Risk Minimization (2019)](https://arxiv.org/abs/1907.02893) |
| RPGNN                | [Relational Pooling for Graph Representations. ICML 2019](https://arxiv.org/abs/1903.02541) |
| GNNs                 | [Graph Convolutional Networks. ICLR 2017](https://arxiv.org/abs/1609.02907) and [Graph Isomorphism Network. ICLR 2019](https://arxiv.org/abs/1810.00826) |
| Transformer (SMILES) | [Transformer (Attention is All You Need. NeurIPS 2017)](https://arxiv.org/abs/1706.03762) based on SMILES strings |
| LSTM (SMILES)        | [Long short-term memory (Neural Computation 1997)](https://ieeexplore.ieee.org/abstract/document/6795963) based on SMILES strings |

### Generative Models

| Model      | Reference           |
|------------|---------------------|
| Graph DiT  | [Graph Diffusion Transformers for Multi-Conditional Molecular Generation. NeurIPS 2024](https://openreview.net/forum?id=cfrDLD1wfO) |
| DiGress    | [DiGress: Discrete Denoising Diffusion for Graph Generation. ICLR 2023](https://openreview.net/forum?id=UaAD-Nu86WX) |
| GDSS       | [Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations. ICML 2022](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf) |
| MolGPT     | [MolGPT: Molecular Generation Using a Transformer-Decoder Model. Journal of Chemical Information and Modeling 2021](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600) |
| JTVAE     | [Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018.](https://proceedings.mlr.press/v80/jin18a) |
| GraphGA    | [A Graph-Based Genetic Algorithm and Its Application to the Multiobjective Evolution of Median Molecules. Journal of Chemical Information and Computer Sciences 2004](https://pubs.acs.org/doi/10.1021/ci034290p) |
| LSTM (SMILES)        | [Long short-term memory (Neural Computation 1997)](https://ieeexplore.ieee.org/abstract/document/6795963) based on SMILES strings |

### Representation Models

| Model        | Reference           |
|--------------|---------------------|
| MoAMa        | [Motif-aware Attribute Masking for Molecular Graph Pre-training. LoG 2024](https://arxiv.org/abs/2309.04589) |
| GraphMAE        | [GraphMAE: Self-Supervised Masked Graph Autoencoders. KDD 2022](https://arxiv.org/abs/2205.10803) |
| AttrMasking  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| ContextPred  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| EdgePred     | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| InfoGraph    | [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. ICLR 2020](https://arxiv.org/abs/1908.01000) |
| Supervised   | Supervised pretraining |
| Pretrained   | More than ten pretrained models from [Hugging Face](https://huggingface.co) |

## Project Structure

See the structure of `torch_molecule` with the command `tree -L 2 torch_molecule -I '__pycache__|*.pyc|*.pyo|.git|old*'`

## Plan

1. **Predictive Models**: Done: GREA, SGIR, IRM, GIN/GCN w/ virtual, DIR. SMILES-based LSTM/Transformers. TODO more
2. **Generative Models**: Done: Graph DiT, GraphGA, DiGress, GDS, MolGPT TODO: more
3. **Representation Models**: Done: MoAMa, AttrMasking, ContextPred, EdgePred. Many pretrained models from HF. TODO: checkpoints, more 

> **Note**: This project is in active development, and features may change.

## Acknowledgements

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.