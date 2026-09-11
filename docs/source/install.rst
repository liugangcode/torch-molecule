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

+----------------------------------------------+----------------------------------------------+
| Model                                        | Required Package                             |
+==============================================+==============================================+
| HFPretrainedMolecularEncoder                 | transformers                                 |
+----------------------------------------------+----------------------------------------------+
| HFPretrainedMolecularGenerator               | transformers                                 |
+----------------------------------------------+----------------------------------------------+
| HFPretrainedMolecularGenerator (MolGen)      | transformers, selfies                         |
+----------------------------------------------+----------------------------------------------+
| HFPretrainedMolecularGenerator (Molexar)     | transformers, fragment-selfies, molexar      |
+----------------------------------------------+----------------------------------------------+
| HFPretrainedMolecularGenerator (SAFE-GPT)    | transformers, safe-mol                       |
+----------------------------------------------+----------------------------------------------+
| BFGNNMolecularPredictor                      | torch-scatter                                |
+----------------------------------------------+----------------------------------------------+
| GRINMolecularPredictor                       | torch-scatter                                |
+----------------------------------------------+----------------------------------------------+

**For models that require** ``transformers``: ``pip install transformers``

**For MolGen** (``selfies``): ``pip install "selfies>=2.1"``. Source: `aspuru-guzik-group/selfies <https://github.com/aspuru-guzik-group/selfies>`_.

**For Molexar:** ``pip install fragment-selfies loguru`` (`Fragment-SELFIES <https://github.com/fairydance/Fragment-SELFIES>`_) and ``pip install git+https://github.com/fairydance/Molexar.git`` (`Molexar <https://github.com/fairydance/Molexar>`_). Molexar itself requires ``transformers>=5.8``.

**For SAFE-GPT:** ``pip install safe-mol`` (`SAFE <https://github.com/datamol-io/safe>`_).
