# torch-molecule 0.2.0

This release adds four public model classes, dataset discovery and loading from
the molecular dataset hub, and new dataset splitting tools. The package now
provides 11 predictors, 9 encoders, and 11 generators.

## Models

- **SAND** learns molecular representations from pairs and similarity labels,
  with dense and quantized embeddings.
- **GenMol** trains masked diffusion over SAFE sequences and supports scaffold
  generation.
- **DemoDiff** trains graph diffusion with optional demonstration molecules and
  normalized similarity scores.
- **HFPretrainedMolecularGenerator** loads pretrained molecular generators,
  supports fine-tuning, and saves models and tokenizers to local directories.

SAND, GenMol, and DemoDiff train from scratch through `fit()`. The Hugging Face
generator loads an existing checkpoint before optional fine-tuning.

## Datasets

- Discover datasets with `list_datasets()` and inspect them with `dataset_info()`.
- Load structures, targets, metadata, and source splits with `load_dataset()`.
- Choose random, scaffold, Butina, or molecular-size train/holdout splits.
- Use explicit target names and a dataset collection release for reproducible
  experiments. Random splitting assigns rows independently, so duplicate
  molecules can appear on both sides.

## Installation

Python 3.10 or newer is required. GenMol and Molexar require Python 3.11 or
newer. Pretrained generators require PyTorch 2.6 or newer. Additional extras
are available for pretrained generators and graph scatter operations. GenMol
and Molexar require separate environments because their Transformers version
requirements conflict.

## Migrating from 0.1.7

The seven convenience dataset loaders remain available with a deprecation
warning. Prefer `load_dataset()` and name the desired targets explicitly:

```python
from torch_molecule.datasets import load_dataset

data = load_dataset("qm9", targets=["gap"], local_dir="torchmol_data")
```

The loaders now use curated collection data. SMILES representations, row counts,
row order, target availability, and default target columns can differ from
0.1.7. Existing downloaded CSV files are not the new Parquet cache format.
`return_local_data_path=True` now returns a cache directory rather than an
individual CSV filename.

`load_gasperm()` now loads `polymer_gas_permeability` from the hub and needs a
download or prepared cache on first use. Import it from `torch_molecule.datasets`;
the old `torch_molecule.datasets.load_local_csv` module is no longer available.
Use `load_dataset("admet_ai", ...)` in place of `load_admet()`.

## Documentation and examples

The documentation includes individual model references, inline examples,
responsive layouts, and automatically refreshed dataset statistics.
Public examples are available in `examples/` in the source archive.
