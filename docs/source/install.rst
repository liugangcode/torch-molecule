.. _install:

Installation
============

To install `torch-molecule`, follow these steps:

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

    Editable install:
    ```bash
    pip install -e .
    ```

3. **Install `torch_molecule` from PyPI**:

    Legacy (PyPI test):
    ```bash
    pip install -i https://test.pypi.org/simple/ torch-molecule
    ```

**Required Dependencies**:

Dependencies are listed in `https://github.com/liugangcode/torch-molecule/requirements.txt`. Example contents:

```text
# Install PyTorch with CUDA 11.8
-f https://download.pytorch.org/whl/cu118/torch_stable.html
torch==2.2.0+cu118

# Install PyTorch Geometric and related packages
-f https://data.pyg.org/whl/torch-2.2.0+cu118.html
torch_geometric==2.6.1
torch_cluster
torch_scatter

# Other dependencies
huggingface_hub
joblib==1.3.2
networkx==3.2.1
pandas==2.2.3
PyYAML==6.0.2
rdkit==2023.9.5
scikit_learn==1.4.1.post1
scipy==1.14.1
tqdm==4.66.2

optuna
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```