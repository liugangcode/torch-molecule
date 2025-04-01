Model Overview
====================================================

The ``torch-molecule`` library provides a unified interface for molecular property prediction, generation and encoding. All models inherit from the :class:`torch_molecule.base.base.BaseModel` class and share common methods for model training, evaluation and persistence.

Model Persistence
^^^^^^^^^^^^^^^^^
- ``load_from_local``: Load a saved model from a local file
- ``save_to_local``: Save the current model to a local file
- ``load_from_huggingface``: Load a model from a Hugging Face repository
- ``push_to_huggingface``: Push the current model to a Hugging Face repository
- ``load``: Load a model from either local storage or Hugging Face
- ``save``: Save the model to either local storage or Hugging Face

(For detailed API documentation of the base class, please refer to :doc:`api/base`.)

.. toctree::
   :maxdepth: 3
   :caption: Molecular Predictor Models

   api/predictor

.. toctree::
   :maxdepth: 3
   :caption: Molecular Generator Models

   api/generator

.. toctree::
   :maxdepth: 3
   :caption: Molecular Encoder Models

   api/encoder
