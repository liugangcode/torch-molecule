# Documentation

Build from the repository root in the project environment:

```bash
micromamba run -n torch-molecule python -m pip install -r docs/requirements.txt
micromamba run -n torch-molecule sphinx-build -b html -W --keep-going docs/source docs/build/html
micromamba run -n torch-molecule python -m http.server 8000 --directory docs/build/html
```

`micromamba run -n torch-molecule make -C docs html` writes to the same
`docs/build/html` directory. Rebuild after changing or pulling documentation;
the HTTP server serves the generated files without rebuilding them.

On a push to `main`, GitHub Actions installs the documentation dependencies,
builds `docs/build/html`, and publishes that directory to the `gh-pages` branch.
Generated HTML can remain ignored: the workflow creates it from the tracked
source files before publishing.

## Dataset statistics

The dataset guide uses the two responsive figures in
`source/_static/dataset-statistics*.svg`. They use the same published `hub.json`
as the molecular dataset hub; no dataset files are downloaded. Refresh them with:

```bash
micromamba run -n torch-molecule python docs/update_dataset_statistics.py
```

Use `--input path/to/hub.json` to render a saved release offline. Ordinary Sphinx
builds use the committed figures and need no network access for the statistics.
The documentation workflow refreshes them on pushes to `main` and daily at
03:00 UTC, then publishes the website. It can also be run manually from GitHub
Actions. The README embeds the published website figures, so both follow the
latest successful deployment without automated commits to `main`. Browser and
GitHub image caches may delay the new image. The image URLs become
available on the first deployment containing these figures.

## Add a model reference

Create `docs/source/api/models/MyMolecularPredictor.rst`:

```rst
MyMolecularPredictor
====================

One sentence explaining what the model does and when to use it.

.. model-reference:: torch_molecule.predictor.my_model.modeling_my_model.MyMolecularPredictor
   :methods: fit, predict
   :example: examples/prediction/my_model.py
```

Use the actual class path and list the public methods to document. `:example:`
is optional and points to a Python file relative to the repository root.
Examples are displayed inline using `literalinclude`; editing the script updates
the page on the next build. Prefer a short, runnable example with realistic inputs
and the returned values. Keep test assertions and benchmarking out of usage examples.

Add the page to the relevant API family's table and toctree in
`docs/source/api/predictor.rst`, `encoder.rst`, or `generator.rst`, and add the
model to `docs/source/models.rst`. Model-specific explanations, notes, figures,
or additional examples can use ordinary reStructuredText before or after the
`model-reference` directive. Existing pages show these patterns.

## Maintain API content

Signatures, parameters, return values, and method summaries come from Python
docstrings through Sphinx `autodoc`, `napoleon`, and `autosummary`. Update them in
the implementation rather than copying them into a reference page. Use NumPy-style
sections: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`, and `References`.
Document shapes, units, defaults, and input constraints when they affect usage.

Dataset functions use `autofunction`. Shared methods live in `api/base.rst`.
Use Sphinx cross-references such as `:class:`, `:meth:`, and `:doc:` for internal
links. Existing class URLs and example anchors should remain stable.

## Change the layout

- [Model template](source/_templates/api/model.rst.jinja): import, constructor,
  inline example, method summary, and method details for every model.
- [Sphinx extension](source/_ext/api_reference.py): the small `model-reference`
  directive and the expandable default column list for dataset loaders.
- [Shared CSS](source/_static/torch-molecule.css): native Sphinx field lists,
  code blocks, tables, callouts, and responsive layout.
- [Page template](source/_templates/page.html): site navigation and footer.

Parameter lists retain Sphinx's native markup. No per-model HTML or JavaScript is
needed. Template and example-file changes are tracked by Sphinx for incremental
rebuilds. The template also works with Sphinx's text builder.

After a change, run the strict build above and inspect a class, a method, and a
dataset page at phone and desktop widths. Check long names, nested parameter
descriptions, return values, examples, and light/dark themes. Keep screenshots,
logs, and other review output under `artifacts/`; the website build stays in
`docs/build/html`.

This structure follows PyTorch's use of [autodoc API pages](https://github.com/pytorch/pytorch/blob/main/docs/source/nn.init.rst)
and [Sphinx templates](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#customizing-templates).
