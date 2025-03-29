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

   index
   install
   example

.. toctree::
   :maxdepth: 3
   :caption: Main Components:

.. toggle:: Predictor
   :hidden:

   .. toctree::
      :maxdepth: 1

      api/predictor

.. toggle:: Generator
   :hidden:

   .. toctree::
      :maxdepth: 1

      api/generator

.. toggle:: Encoder
   :hidden:

   .. toctree::
      :maxdepth: 1

      api/encoder

.. toctree::
   :maxdepth: 2
   :caption: API for utilities:

   .. toggle:: Utils
      :hidden:

      .. toctree::
         :maxdepth: 1

   api/utils

   .. toggle:: NN
      :hidden:

      .. toctree::
         :maxdepth: 1

         api/nn

   .. toggle:: Base
      :hidden:

      .. toctree::
         :maxdepth: 1

         api/base

.. toctree::
   :maxdepth: 2
   :caption: Additional Information:

   additional
