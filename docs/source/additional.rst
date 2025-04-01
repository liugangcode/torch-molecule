Project Overview
----------------

``torch-molecule`` is a package under active development to support molecular discovery using deep learning. It provides a simple, ``sklearn``-style interface and model checkpoints for fast deployment and benchmarking.

Main components:

1. **Predictive Models**  
   - ✔ GREA, SGIR, IRM, GIN/GCN w/ virtual, DIR
   - ✔ SMILES-based LSTM/Transformers
   - ⏳ More models

2. **Generative Models**  
   - ✔ Graph DiT, GraphGA, DiGress, MolGPT
   - ⏳ GDSS and more

3. **Representation Models**  
   - ✔ MoAMa, AttrMasking, ContextPred, EdgePred  
   - ⏳ more models and pretrained checkpoints

.. note::

   This project is in active development. Interfaces and features may change.

Project Structure
-----------------

.. code-block:: text

   torch_molecule
   ├── base
   ├── encoder
   ├── generator
   ├── nn
   ├── predictor
   └── utils

Acknowledgements
----------------

This project was adapted from `python-project-template <https://github.com/lwaekfjlk/python-project-template>`_.

