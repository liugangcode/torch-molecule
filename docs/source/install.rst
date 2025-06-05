Installation
============

This document explains how to install ``torch-molecule`` and any extra packages you may need.

Installation Steps
------------------

Follow these steps in order. 

Create a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule

a. Install via pip
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch-molecule

b. Install from Source (Latest Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/liugangcode/torch-molecule
   cd torch-molecule

Then install:

.. code-block:: bash

   pip install .

c. Editable Installation for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work on the code directly, install in "editable" mode.

1. Clone the repository (if you have not already):

   .. code-block:: bash

      git clone https://github.com/liugangcode/torch-molecule
      cd torch-molecule

2. Install the dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Install in editable mode:

   .. code-block:: bash

      pip install -e .

Additional Packages
-------------------

Some models require extra libraries. Install these packages if you use the corresponding model:

+------------------------------+-------------------+
| Model                        | Required Package  |
+==============================+===================+
| HFPretrainedMolecularEncoder | transformers      |
+------------------------------+-------------------+
| BFGNNMolecularPredictor      | torch-scatter     |
+------------------------------+-------------------+
| GRINMolecularPredictor       | torch-scatter     |
+------------------------------+-------------------+
