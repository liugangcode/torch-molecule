Project Overview
----------------

``torch-molecule`` is a package under active development to support molecular discovery using deep learning. It provides a simple, ``sklearn``-style interface and model checkpoints for fast deployment and benchmarking.

Main components:

1. **Predictive Models**
   - ✔ GREA, SGIR, IRM, GIN/GCN w/ virtual, DIR, SSR
   - ✔ GRIN, BFGNN, RPGNN
   - ✔ SMILES-based LSTM / Transformers
   - ⏳ More models

2. **Generative Models**
   - ✔ Graph DiT, GraphGA, DiGress, GDSS, JTVAE, MolGPT, LSTM
   - ✔ DeFoG
   - ✔ Hugging Face pretrained generators: NovoMolGen, MolGen, Molexar, SAFE-GPT
   - ⏳ More models

3. **Representation Models**
   - ✔ MoAMa, GraphMAE, AttrMasking, ContextPred, EdgePred, InfoGraph, Supervised
   - ✔ Hugging Face pretrained encoders (ChemBERTa, ChemGPT, and related checkpoints)
   - ⏳ More models and pretrained checkpoints

.. note::

   This project is in active development. Interfaces and features may change.

Project Structure
-----------------

.. code-block:: text

   torch_molecule
   ├── base
   ├── datasets
   ├── encoder
   ├── generator
   ├── nn
   ├── predictor
   └── utils

Acknowledgements
----------------

This project was adapted from `python-project-template <https://github.com/lwaekfjlk/python-project-template>`_.

Contributors
~~~~~~~~~~~~

- **Man Hei Matthew Thom**: Pretrained Generator (NovoMolGen, MolGen, Molexar, SAFE-GPT), dataset splitting modules
- **Eric Inae**: MoAMa, GraphMAE, AttrMasking, ContextPred, EdgePred
- **Yihan Zhu**: DeFoG, GRIN, BFGNN, RPGNN, Transformer (SMILES)
