.. image:: ../../assets/logo.png
   :align: center

torch-molecule documentation
============================

Welcome to the torch-molecule documentation. torch-molecule is an actively developed package designed to facilitate molecular discovery through deep learning. It offers a user-friendly interface similar to `sklearn`, and includes model checkpoints for efficient deployment and benchmarking across various molecular tasks. The package currently focuses on three main components:

- Predictors: Molecular property prediction models that make predictions (regression or classification) based on molecular graphs.
- Generators: Molecular graph generators that generate new molecules.
- Encoders: Molecular graph encoders that convert molecular graphs into a vector representation.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   install
   example

.. toctree::
   :maxdepth: 3
   :caption: Main Components:

   api/predictor
   api/generator
   api/encoder

.. toctree::
   :maxdepth: 2
   :caption: API for utilities:

   api/utils
   api/nn
   api/base

.. toctree::
   :maxdepth: 2
   :caption: Additional Information:

   additional
