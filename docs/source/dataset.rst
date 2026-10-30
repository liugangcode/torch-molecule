Datasets
========

Model inputs are lists of SMILES strings. For supervised prediction, add a
numerical target array with shape ``(n_molecules, n_tasks)``. Dataset loaders
return a :class:`~torch_molecule.datasets.constant.SMILESDataset` with ``data``
and ``target`` attributes; unlabeled datasets have ``target=None``.

Collection statistics
---------------------

Explore small-molecule and polymer datasets in the
`molecular dataset hub <https://liugangcode.github.io/molecular-dataset-hub/>`_,
including their original sources, licenses, quality tags, and available data.

.. raw:: html

   <a class="tm-dataset-statistics" href="https://liugangcode.github.io/molecular-dataset-hub/#explore">
     <picture>
       <source media="(max-width: 600px)" srcset="_static/dataset-statistics-compact.svg">
       <img src="_static/dataset-statistics.svg" alt="Open the Molecular Dataset Hub for current dataset counts, Hugging Face coverage, source rows, tasks, and the interactive map.">
     </picture>
   </a>

Read the hub's `current counts <https://liugangcode.github.io/molecular-dataset-hub/#stats>`_
or select the figure to explore its interactive map.
Hugging Face entries include public and gated datasets, and some provide files
instead of SMILES tables. Source rows can repeat structures and are not a count
of unique molecules. Each dataset retains its own license.

Load a dataset
--------------

``load_dataset`` loads any dataset of the
`molecular dataset hub <https://liugangcode.github.io/molecular-dataset-hub/>`_
by name, the way scikit-learn's loaders do:

.. code-block:: python

   from torch_molecule.datasets import load_dataset

   X, y = load_dataset("qm9", targets=["gap"], return_X_y=True)

Without ``return_X_y`` you get a
:class:`~torch_molecule.datasets.constant.SMILESDataset` that also says what the
numbers are:

.. code-block:: python

   data = load_dataset("qm9", targets=["homo", "lumo", "gap"])
   data.data[:2], data.target.shape     # SMILES, (133885, 3)
   data.info["targets"]["gap"]["units"] # 'Hartree'

Some datasets hold several tables, one per target, assay, or property; name one
as the second argument. MoleculeACE has one per protein target:

.. code-block:: python

   data = load_dataset("moleculeace", "chembl204_ki")
   data.info["title"]                                  # 'Prothrombin (CHEMBL204) · Ki'
   data.info["targets"]["y [pEC50/pKi]"]["about"]      # ChEMBL target, organism, UniProt

``list_datasets()`` lists the hub, optionally by text, category, or quality
grade; ``dataset_info(name)`` describes one dataset and its configs:

.. code-block:: python

   from torch_molecule.datasets import dataset_info, list_datasets

   list_datasets("thrombin")                    # finds moleculeace by a config title
   list_datasets(category="adme", grade="ready")
   [t["config"] for t in dataset_info("moleculeace")["tables"]]

Labels stay as the source reported them: missing labels are ``NaN`` (never 0),
and censored values such as ``<10`` are ``NaN`` unless ``include_censored=True``.
Datasets that cannot be redistributed are listed too; loading one raises an
error that names the original source. Gated datasets need
``huggingface-cli login`` and an accepted access request on the dataset page.
Files download on first use to ``~/torchmol_data`` (set ``TORCHMOL_DATA`` or
``local_dir=`` to change it); ``release="v2026.10.31"`` pins a hub release.
Names from earlier releases, such as ``moleculeace_chembl204_ki``, still load.

.. note::

   This interface is newer than the 0.2.0 release on PyPI. Until the next
   release, install from GitHub:
   ``pip install git+https://github.com/liugangcode/torch-molecule``.

Bring your own data
-------------------

Pass SMILES directly to a model, or wrap them in ``SMILESDataset`` to use the
built-in splitting tools. For an unlabeled CSV with a ``smiles`` column:

.. code-block:: python

   import csv
   from torch_molecule.datasets import SMILESDataset

   with open("data/molecule100.csv", newline="") as handle:
       smiles = [row["smiles"] for row in csv.DictReader(handle)]

   data = SMILESDataset(data=smiles, target=None)

The source checkout includes ``data/molecule100.csv`` and ``data/polymer100.csv``
for small local examples. Both contain SMILES; the polymer dataset uses ``*``
attachment atoms. Check the selected model's support for polymer inputs and
attachment atoms before using it for polymer generation.

Choose a split
--------------

When the source defines a split, load it by name, so results compare with
published ones. A test split drops molecules that also appear in training:

.. code-block:: python

   train = load_dataset("moleculeace", "chembl204_ki", split="train")
   test = load_dataset("moleculeace", "chembl204_ki", split="test")

Some benchmarks provide several splits, for example five scaffold and five
random splits in WelQrate. ``scheme`` picks one; the default is the first:

.. code-block:: python

   for scheme in ["scaffold_1", "scaffold_2", "scaffold_3"]:
       train = load_dataset("welqrate", "aid1798", split="train", scheme=scheme)
       test = load_dataset("welqrate", "aid1798", split="test", scheme=scheme)

Otherwise split the dataset yourself:

.. code-block:: python

   data = load_dataset("polymer_gas_permeability", targets=["CO2"])
   train, validation = data.train_test_split(
       test_size=0.2,
       method="random",
       seed=42,
   )

Pass ``train.data`` and ``train.target`` to ``fit``; supply the validation data
through ``X_val`` and ``y_val`` when the model supports them. Keep a separate
test set for a final performance estimate. ``data.groups`` holds a structure key
per row, but the splitters preserve this metadata without using it to assign
rows. Random splitting assigns individual rows, so repeated structures can
occur in both splits. Use ``scaffold`` or ``butina``, or split by group yourself, when
repeated measurements must stay together.

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Method
     - Holdout strategy
     - Useful for
   * - ``random``
     - Randomly assign rows to each split; repeated structures may overlap
     - An initial baseline
   * - ``scaffold``
     - Keep Bemis–Murcko scaffold groups separate
     - Testing on unseen molecular scaffolds
   * - ``butina``
     - Separate clusters based on Morgan fingerprint Tanimoto similarity
     - Testing on different structural neighborhoods
   * - ``size``
     - Split by heavy-atom count
     - Testing transfer across molecular sizes

Use ``subsample(n=100, seed=42)`` for a quick local experiment. Evaluate
benchmarks on the intended dataset and split rather than a debugging subset.

More than SMILES and numbers
----------------------------

``columns`` returns other source columns next to the labels, row for row, in
``data.extra``; they follow the rows through ``train_test_split``:

.. code-block:: python

   data = load_dataset("welqrate", "aid1798", columns=["CID"])
   data.extra["CID"][:3]           # PubChem compound IDs

Some tables hold one row per measurement, with a task column such as an assay
ID. ``task_values`` lists the tasks, most-measured first; each one you pass
becomes a target column:

.. code-block:: python

   from torch_molecule.datasets import task_values

   assays = task_values("cara", "lo_all")
   data = load_dataset("cara", "lo_all", task_values=assays[:10])

For tables that are not SMILES and numbers (question and answer text,
reactions, mixtures), ``load_table`` returns every column as a
``pyarrow.Table``; ``download_files`` fetches native files such as 3D
structures, spectra and images:

.. code-block:: python

   from torch_molecule.datasets import download_files, load_table

   questions = load_table("fgbench", split="test").to_pandas()
   paths = download_files("qm9", patterns=["structures/*"])

Loader reference
----------------

.. autofunction:: torch_molecule.datasets.load_dataset
.. autofunction:: torch_molecule.datasets.list_datasets
.. autofunction:: torch_molecule.datasets.dataset_info
.. autofunction:: torch_molecule.datasets.task_values
.. autofunction:: torch_molecule.datasets.load_table
.. autofunction:: torch_molecule.datasets.download_files

Deprecated loaders
------------------

These functions still work but call ``load_dataset`` and will be removed. Rows
come from the collection's clean tables: canonical SMILES, unparseable
structures removed, identical rows collapsed.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Replacement
   * - :func:`load_qm9 <torch_molecule.datasets.load_qm9>`
     - ``load_dataset("qm9", targets=["gap"])``
   * - :func:`load_chembl2k <torch_molecule.datasets.load_chembl2k>`
     - ``load_dataset("chembl2k")``
   * - :func:`load_broad6k <torch_molecule.datasets.load_broad6k>`
     - ``load_dataset("broad6k")``
   * - :func:`load_toxcast <torch_molecule.datasets.load_toxcast>`
     - ``load_dataset("toxcast")``
   * - :func:`load_admet <torch_molecule.datasets.load_admet>`
     - ``load_dataset("admet_ai")``
   * - :func:`load_gasperm <torch_molecule.datasets.load_gasperm>`
     - ``load_dataset("polymer_gas_permeability")``
   * - :func:`load_zinc250k <torch_molecule.datasets.load_zinc250k>`
     - ``load_dataset("zinc250k", targets=[])``

.. toctree::
   :hidden:

   api/datasets/load_qm9
   api/datasets/load_chembl2k
   api/datasets/load_broad6k
   api/datasets/load_toxcast
   api/datasets/load_admet
   api/datasets/load_gasperm
   api/datasets/load_zinc250k

Dataset interface
-----------------

.. autoclass:: torch_molecule.datasets.constant.SMILESDataset
   :members: subsample, train_test_split

.. raw:: html

   <a class="tm-api-redirect" id="torch_molecule.datasets.load_qm9" href="api/datasets/load_qm9.html#torch_molecule.datasets.load_qm9">load_qm9 reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_chembl2k" href="api/datasets/load_chembl2k.html#torch_molecule.datasets.load_chembl2k">load_chembl2k reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_broad6k" href="api/datasets/load_broad6k.html#torch_molecule.datasets.load_broad6k">load_broad6k reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_toxcast" href="api/datasets/load_toxcast.html#torch_molecule.datasets.load_toxcast">load_toxcast reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_admet" href="api/datasets/load_admet.html#torch_molecule.datasets.load_admet">load_admet reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_gasperm" href="api/datasets/load_gasperm.html#torch_molecule.datasets.load_gasperm">load_gasperm reference</a>
   <a class="tm-api-redirect" id="torch_molecule.datasets.load_zinc250k" href="api/datasets/load_zinc250k.html#torch_molecule.datasets.load_zinc250k">load_zinc250k reference</a>
