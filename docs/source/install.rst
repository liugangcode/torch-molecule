Installation
============

Create a dedicated environment, then install the published package or the latest
source. Python 3.12 is a practical starting point for the current models.

Create an environment
---------------------

.. code-block:: bash

   micromamba create -n torch-molecule -c conda-forge python=3.12 pip
   eval "$(micromamba shell hook --shell bash)"
   micromamba activate torch-molecule

An existing project environment can be reused. If you use Mamba, use
``mamba create`` and activate it with your configured Conda shell.

Install the package
-------------------

For the latest published release:

.. code-block:: bash

   python -m pip install torch-molecule

Install from source
-------------------

For the current source, including recently added models:

.. code-block:: bash

   git clone https://github.com/liugangcode/torch-molecule.git
   cd torch-molecule
   python -m pip install -e .

Check the installation:

.. code-block:: bash

   python -c "import torch_molecule; print(torch_molecule.__version__)"

Then try :doc:`example` or choose a model in the :doc:`overview`.

GPU support
-----------

Models accept ``device="cpu"``, ``device="cuda"``, or a specific GPU such as
``device="cuda:1"``. When omitted, supported models choose CUDA if available.
Install a PyTorch build suitable for your system using the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Optional model packages
-----------------------

Install additional packages only for the models you use. SAND and DemoDiff use
the core project dependencies.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Model
     - Additional packages
   * - Hugging Face pretrained encoders and generators
     - ``transformers``; checkpoint-specific packages may also be needed
   * - MolGen through the pretrained generator
     - ``transformers``, ``selfies>=2.1``
   * - Molexar through the pretrained generator
     - ``transformers``, ``fragment-selfies``, ``molexar``
   * - SAFE-GPT through the pretrained generator
     - ``transformers``, ``safe-mol``
   * - GenMol
     - ``transformers==4.57.1``, ``safe-mol>=0.2.1,<0.3``
   * - BFGNN and GRIN
     - ``torch-scatter``
   * - GRIN with ``repetition_augmentation=True``
     - ``CombineMols`` in addition to ``torch-scatter``

**GenMol.** From the source checkout, install the supported extra
(Python 3.11 or newer):

.. code-block:: bash

   python -m pip install -e ".[genmol]"

**Hugging Face models.** Install the packages required by your chosen checkpoint.
For example, SAFE-GPT uses:

.. code-block:: bash

   python -m pip install transformers safe-mol

For Molexar, follow the package instructions in the
`Molexar repository <https://github.com/fairydance/Molexar>`_ and
`Fragment-SELFIES repository <https://github.com/fairydance/Fragment-SELFIES>`_.
Use a separate environment when checkpoint dependencies conflict with another
model's version requirements.

**Graph scatter operations.** Select a ``torch-scatter`` wheel matching your
installed PyTorch and CUDA versions using the
`PyTorch Geometric installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_.

Build the documentation
-----------------------

From the repository root, inside the project environment:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html docs/source artifacts/docs/html
   python -m http.server 8000 --directory artifacts/docs/html

Open ``http://localhost:8000`` to browse the local documentation.
