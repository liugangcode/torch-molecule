About the project
=================

``torch-molecule`` brings molecular property prediction, representation learning,
and generation into a shared Python interface. It combines PyTorch models with
SMILES inputs, training utilities, dataset loaders, and checkpoint persistence.

The project is developed openly on
`GitHub <https://github.com/liugangcode/torch-molecule>`_. Browse the
:doc:`overview` for supported workflows or read the class references for method
papers and original implementations.

Contribute
----------

`Report an issue <https://github.com/liugangcode/torch-molecule/issues>`_ with a
small reproducible example, your package version, and the relevant error output.
Model integrations, documentation improvements, and additional examples are
welcome through pull requests.

For a local development installation:

.. code-block:: bash

   micromamba activate torch-molecule
   python -m pip install -e ".[dev]"

Keep method-specific code with its model, follow comparable classes, and add
focused tests and a usable example. See :doc:`install` for building the docs.

Contributors
------------

The following contributors have added model families and data utilities:

* **Man Hei Matthew Thom** — pretrained generators and dataset splitting.
* **Eric Inae** — Moama, GraphMAE, AttrMask, ContextPred, and EdgePred.
* **Yihan Zhu** — DeFoG, GRIN, BFGNN, RPGNN, and the SMILES Transformer.

See the
`repository contributors <https://github.com/liugangcode/torch-molecule/graphs/contributors>`_
for the current contributor list.

License and acknowledgements
----------------------------

The project is distributed under the
`MIT license <https://github.com/liugangcode/torch-molecule/blob/main/LICENSE>`_.
Integrated methods retain their accompanying licenses and attribution.
The initial project structure was adapted from
`python-project-template <https://github.com/lwaekfjlk/python-project-template>`_.
