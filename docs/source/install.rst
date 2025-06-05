Installation
============

To install `torch-molecule`, follow these steps:

1. **Create a Conda environment**:
   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install using pip (0.1.0)**:

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

4. **Install editable `torch_molecule` (for developer)**:

   Clone the repository:
   
   .. code-block:: bash

      git clone https://github.com/liugangcode/torch-molecule

   Install the requirements:

   .. code-block:: bash

      pip install -r requirements.txt

   Editable install:
   
   .. code-block:: bash

      pip install -e .


### Additional Packages

| Model | Required Packages |
|-------|-------------------|
| HFPretrainedMolecularEncoder | transformers |
| BFGNNMolecularPredictor | torch-scatter |
| GRINMolecularPredictor | torch-scatter |