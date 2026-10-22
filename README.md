<p align="center">
  <img src="https://raw.githubusercontent.com/liugangcode/torch-molecule/main/assets/logo.png" alt="torch-molecule logo" width="600"/>
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

`torch-molecule` is a package that facilitates molecular discovery through deep learning, featuring a user-friendly, `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. The package focuses on three main components: **Predictive Models**, **Generative Models**, and **Representation Models**, which make molecular AI models easy to implement and deploy. 

<p align="center">
  <img src="https://raw.githubusercontent.com/liugangcode/torch-molecule/main/assets/cover.png" alt="scikit-learn vs torch-molecule comparison" width="800"/>
</p>

See the [List of Supported Models](#list-of-supported-models) section for all available models.

## Installation

Python 3.10 or newer is required; GenMol and Molexar require Python 3.11 or newer.

1. **Create a mamba environment**:
   ```bash
   micromamba create --name torch-molecule --channel conda-forge python=3.12 pip
   eval "$(micromamba shell hook --shell bash)"
   micromamba activate torch-molecule
   ```

2. **Install using pip**:
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

Optional model requirements install as pip extras, e.g. `pip install "torch-molecule[molgen]"` (or `pip install ".[molgen]"` from a source checkout):

| Extra | Installs | For |
|-------|----------|-----|
| `hf` | torch>=2.6, transformers | HFPretrainedMolecularEncoder, HFPretrainedMolecularGenerator |
| `molgen` | torch>=2.6, transformers, selfies>=2.1 | MolGen |
| `molexar` | torch>=2.6, transformers>=5.8, loguru | Molexar (Python 3.11+); also install [Fragment-SELFIES](https://github.com/yiyanliao/Fragment-SELFIES) and [Molexar](https://github.com/yiyanliao/Molexar) from source |
| `safe` | torch>=2.6, transformers, safe-mol | SAFE-GPT |
| `genmol` | transformers==4.57.1, safe-mol>=0.2.1,<0.3 | GenMol (Python 3.11+) |
| `scatter` | torch-scatter | BFGNN, GRIN |
| `grin` | torch-scatter, CombineMols | GRIN with `repetition_augmentation=True` |

Hugging Face pretrained generators require PyTorch 2.6 or newer.

Use separate environments for GenMol and Molexar: GenMol requires
`transformers==4.57.1`, while Molexar requires `transformers>=5.8`.

After installing the `molexar` extra, install its source packages:

```bash
python -m pip install git+https://github.com/yiyanliao/Fragment-SELFIES.git git+https://github.com/yiyanliao/Molexar.git
```

Install a `torch-scatter` wheel matching your PyTorch and CUDA versions using
the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```python
from torch_molecule import HFPretrainedMolecularGenerator

model = HFPretrainedMolecularGenerator(
    repo_id="datamol-io/safe-gpt",
)
model.fit()
print(model.generate(n_samples=5))
print(model.generate(n_samples=5, scaffold="c1ccccc1"))
```

## Usage

> More examples can be found in the `examples` folder.

### Datasets

Explore small-molecule and polymer datasets in the [molecular dataset hub](https://liugangcode.github.io/molecular-dataset-hub/), with original sources, licenses, units, source splits, and quality tags.

See the [dataset documentation](https://liugangcode.github.io/torch-molecule/dataset.html)
for dataset selection, caching, target columns, and train/test splits.

<a href="https://liugangcode.github.io/molecular-dataset-hub/#explore">
  <picture>
    <source media="(max-width: 600px)" srcset="https://liugangcode.github.io/torch-molecule/_static/dataset-statistics-compact.svg">
    <img src="https://liugangcode.github.io/torch-molecule/_static/dataset-statistics.svg" alt="Open the Molecular Dataset Hub for current dataset counts, Hugging Face coverage, source rows, tasks, and the interactive map.">
  </picture>
</a>

[Current counts](https://liugangcode.github.io/molecular-dataset-hub/#stats) follow the hub's published metadata. Hugging Face entries include public and gated datasets and files without SMILES tables. Source rows may repeat structures; each dataset retains its own license.

Load published SMILES tables by their original dataset name:

```python
from torch_molecule.datasets import list_datasets, load_dataset

list_datasets(category="adme", grade="ready")      # browse by category, quality grade, or license

data = load_dataset("qm9", targets=["homo", "lumo", "gap"], local_dir="torchmol_data")
smiles_list, property_np_array = data.data, data.target   # 133,885 molecules, shape (133885, 3)
data.info["targets"]["gap"]["units"]                        # 'Hartree'

train = load_dataset("moleculeace_chembl204_ki", split="train")   # source splits where defined;
test = load_dataset("moleculeace_chembl204_ki", split="test")     # test drops molecules seen in train

zinc = load_dataset("zinc250k", targets=[])                 # structures only: target is None
```

`load_qm9`, `load_chembl2k`, `load_broad6k`, `load_toxcast`, `load_admet`, `load_gasperm`, and `load_zinc250k` still work but are deprecated aliases of `load_dataset` and will be removed.

(We are actively adding more datasets. We welcome your suggestions and contributions on your datasets!)

### Fit a Model

After preparing the dataset, split it, then fit a model with an sklearn-style API (no extra SMILES featurization is required):

```python
from torch_molecule.datasets import load_dataset
from torch_molecule import GREAMolecularPredictor

data = load_dataset("qm9", targets=["gap"], local_dir="torchmol_data")
# "random" | "scaffold" | "butina" | "size"
# scaffold: unseen Bemis-Murcko scaffolds; butina: Tanimoto clusters; size: heavy-atom count
train, val = data.train_test_split(test_size=0.2, method="scaffold", seed=42)

grea = GREAMolecularPredictor(
    num_task=1,
    task_type="regression",
    evaluate_higher_better=False,
    verbose="progress_bar" #or "print_statement" recommended for jupyter notebooks, or "none"
)

# Fit with automatic hyperparameter tuning with 10 attempts, or implement .fit() with the default/manual hyperparameters
grea.autofit(
    X_train=train.data,
    y_train=train.target,
    X_val=val.data,
    y_val=val.target,
    n_trials=10,
)
```

Random splitting assigns individual rows, so repeated structures can occur
in both splits. The splitters preserve `data.groups` metadata but do not use it
to assign rows. Use `scaffold` or `butina`, or split by group yourself, when
repeated measurements must stay together.

### Checkpoints

`torch-molecule` provides checkpoint functions that can be interacted with on Hugging Face:

```python
from torch_molecule import GREAMolecularPredictor

repo_id = "user/repo_id"  # replace with your own Hugging Face username and repo_id

# Save the trained model to Hugging Face
grea.save_to_hf(
    repo_id=repo_id,
    task_id="qm9_grea",
    commit_message="Upload qm9_grea",
    private=False
)

# Load a pretrained checkpoint from Hugging Face
model = GREAMolecularPredictor()
model.load_from_hf(repo_id=repo_id, local_cache="GREAMolecularPredictor.pt")

# Adjust model parameters and make predictions
model.set_params(verbose='none')
predictions = model.predict(smiles_list)
```

Or you can save the model to a local path:

```python
grea.save_to_local("qm9_grea.pt")

new_model = GREAMolecularPredictor()
new_model.load_from_local("qm9_grea.pt")
```

## List of Supported Models

### Predictive Models

| Model                | Reference           |
|----------------------|---------------------|
| GRIN                 | [Learning Repetition-Invariant Representations for Polymer Informatics. NeurIPS 2025.](https://arxiv.org/abs/2505.10726) |
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
| DemoDiff   | [Graph Diffusion Transformers are In-Context Molecular Designers. ICLR 2026](https://openreview.net/forum?id=lJ87GN5zJc) |
| GenMol     | [GenMol: A Drug Discovery Generalist with Discrete Diffusion. ICML 2025](https://proceedings.mlr.press/v267/lee25o.html) |
| DeFoG      | [DeFoG: Discrete Flow Matching for Graph Generation. ICML 2025](https://openreview.net/pdf?id=KPRIwWhqAZ) |
| Graph DiT  | [Graph Diffusion Transformers for Multi-Conditional Molecular Generation. NeurIPS 2024](https://openreview.net/forum?id=cfrDLD1wfO) |
| DiGress    | [DiGress: Discrete Denoising Diffusion for Graph Generation. ICLR 2023](https://openreview.net/forum?id=UaAD-Nu86WX) |
| GDSS       | [Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations. ICML 2022](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf) |
| MolGPT     | [MolGPT: Molecular Generation Using a Transformer-Decoder Model. Journal of Chemical Information and Modeling 2021](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600) |
| JTVAE      | [Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018.](https://proceedings.mlr.press/v80/jin18a) |
| GraphGA    | [A Graph-Based Genetic Algorithm and Its Application to the Multiobjective Evolution of Median Molecules. Journal of Chemical Information and Computer Sciences 2004](https://pubs.acs.org/doi/10.1021/ci034290p) |
| LSTM (SMILES)        | [Long short-term memory (Neural Computation 1997)](https://ieeexplore.ieee.org/abstract/document/6795963) based on SMILES strings |
| Pretrained | [NovoMolGen](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_BPE): Causal LM pretrained on ZINC-22 for de novo SMILES generation. <br> [MolGen-large](https://huggingface.co/zjunlp/MolGen-large): Seq2Seq SELFIES generator with high chemical validity. <br> [MolGen-large-opt](https://huggingface.co/zjunlp/MolGen-large-opt): MolGen-large fine-tuned for QED / p-logP optimization. <br> [Molexar-10M-base](https://huggingface.co/fairydance/molexar-10m-base): Fragment-SELFIES de novo and fragment-constrained generation. <br> [Molexar-10M-omni](https://huggingface.co/fairydance/molexar-10m-omni): Multi-condition Molexar model for property-guided generation. <br> [SAFE-GPT](https://huggingface.co/datamol-io/safe-gpt): GPT-2 causal LM pretrained on SAFE strings for de novo generation and scaffold-prefix completion. |

### Representation Models

| Model        | Reference           |
|--------------|---------------------|
| SAND | [Learning Compressed Shape-Aware Molecular Representations for Virtual Screening. ICML 2026](https://proceedings.mlr.press/v306/winter26a.html) |
| MoAMa        | [Motif-aware Attribute Masking for Molecular Graph Pre-training. LoG 2024](https://arxiv.org/abs/2309.04589) |
| GraphMAE        | [GraphMAE: Self-Supervised Masked Graph Autoencoders. KDD 2022](https://arxiv.org/abs/2205.10803) |
| AttrMasking  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| ContextPred  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| EdgePred     | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| InfoGraph    | [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. ICLR 2020](https://arxiv.org/abs/1908.01000) |
| Supervised   | Supervised pretraining |
| Pretrained   | [GPT2-ZINC-87M](https://huggingface.co/entropy/gpt2_zinc_87m): GPT-2 based model (87M parameters) pretrained on ZINC dataset with ~480M SMILES strings. <br> [RoBERTa-ZINC-480M](https://huggingface.co/entropy/roberta_zinc_480m): RoBERTa based model (102M parameters) pretrained on ZINC dataset with ~480M SMILES strings. <br> [UniKi/bert-base-smiles](https://huggingface.co/unikei/bert-base-smiles): BERT model pretrained on SMILES strings. <br> [ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1): RoBERTa model pretrained on ZINC dataset with ~100k SMILES strings. <br> ChemBERTa series: Available in multiple sizes and training objectives (MLM/MTR). [ChemBERTa-5M-MLM](https://huggingface.co/DeepChem/ChemBERTa-5M-MLM), [ChemBERTa-5M-MTR](https://huggingface.co/DeepChem/ChemBERTa-5M-MTR), [ChemBERTa-10M-MLM](https://huggingface.co/DeepChem/ChemBERTa-10M-MLM), [ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR), [ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR). <br> ChemGPT series: GPT-Neo based models pretrained on PubChem10M dataset with SELFIES strings. [ChemGPT-1.2B](https://huggingface.co/ncfrey/ChemGPT-1.2B), [ChemGPT-4.7M](https://huggingface.co/ncfrey/ChemGPT-4.7M), [ChemGPT-19M](https://huggingface.co/ncfrey/ChemGPT-19M). |

## Acknowledgements

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.

This project is initiated and maintained by [Gang Liu](https://liugangcode.github.io/) and [Meng Jiang](http://www.meng-jiang.com/), and would not be possible without the valuable contributions from the open-source community, including the following members:

<table>
  <tr>
    <td align="center" width="180px">
      <a href="https://m21hm9.github.io/">
        <img src="https://avatars.githubusercontent.com/u/205813056?v=4" width="100px;" alt="Man Hei Matthew Thom"/><br />
        <sub><b>Man Hei Matthew Thom</b></sub>
      </a>
      <br />
      <sub>Integrating Pretrained Generator (NovoMolGen, MolGen, Molexar, SAFE-GPT), dataset splitting modules</sub>
    </td>
    <td align="center" width="180px">
      <a href="https://scholar.google.com/citations?user=RKiV3FAAAAAJ&hl=en">
        <img src="https://avatars.githubusercontent.com/u/122834595?v=4" width="100px;" alt="Eric Inae"/><br />
        <sub><b>Eric Inae</b></sub>
      </a>
      <br />
      <sub>Integrating MoAMa, GraphMAE, AttrMasking, ContextPred, EdgePred</sub>
    </td>
    <td align="center" width="180px">
      <a href="https://yihan226.github.io/">
        <img src="https://avatars.githubusercontent.com/u/201283026?v=4" width="100px;" alt="Yihan Zhu"/><br />
        <sub><b>Yihan Zhu</b></sub>
      </a>
      <br />
      <sub>Integrating DeFoG, GRIN, BFGNN, RPGNN, Transformer (SMILES)</sub>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/monics-hub">
        <img src="https://avatars.githubusercontent.com/u/148586659?v=4" width="100px;" alt="Monica C. S"/><br />
        <sub><b>Monica C. S</b></sub>
      </a>
      <br />
      <sub>Debugging and integrating training logs (<a href="https://github.com/liugangcode/torch-molecule/pull/19">#19</a>)</sub>
    </td>
  </tr>
</table>
