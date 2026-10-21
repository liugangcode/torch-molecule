"""Share model page layouts and abbreviate long dataset defaults."""

import inspect
from pathlib import Path
from pprint import pformat

from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.template import SphinxRenderer


class ModelReference(SphinxDirective):
    """Render a model's API and example through the shared RST template."""

    required_arguments = 1
    option_spec = {
        "methods": directives.unchanged_required,
        "example": directives.path,
    }

    def run(self):
        template = Path(self.env.srcdir) / "_templates" / "api" / "model.rst.jinja"
        self.env.note_dependency(str(template))
        fullname = self.arguments[0]
        text = SphinxRenderer.render_from_file(template, {
            "fullname": fullname,
            "name": fullname.rsplit(".", 1)[-1],
            "methods": [name.strip() for name in self.options["methods"].split(",")],
            "example": self.options.get("example"),
        })
        self.state_machine.insert_input(text.splitlines(), str(template))
        return []


class AbbreviatedDefault:
    def __repr__(self):
        return "..."


def long_target_columns(name, obj):
    if not name.startswith("torch_molecule.datasets.load_"):
        return None
    parameter = inspect.signature(obj).parameters.get("target_cols")
    if parameter and isinstance(parameter.default, list) and len(parameter.default) > 8:
        return parameter.default
    return None


def compact_signature(app, what, name, obj, options, signature, return_annotation):
    if long_target_columns(name, obj) is None:
        return None
    parameters = [
        parameter.replace(
            annotation=inspect.Parameter.empty,
            default=AbbreviatedDefault() if parameter.name == "target_cols" else parameter.default,
        )
        for parameter in inspect.signature(obj).parameters.values()
    ]
    return str(inspect.Signature(parameters)), None


def document_defaults(app, what, name, obj, options, lines):
    columns = long_target_columns(name, obj)
    if columns is None:
        return
    for index, line in enumerate(lines):
        if line.strip().startswith("List of target column names"):
            lines[index] = (
                f"        Target columns in output order. Defaults to all {len(columns)} "
                "columns listed below."
            )
    lines.extend([
        "", "The ``...`` in the signature abbreviates the default column list; "
        "omit ``target_cols`` to use it, or pass a list of column names.", "",
        ".. raw:: html", "",
        '   <details class="tm-defaults">',
        f"   <summary>Default target columns ({len(columns)})</summary>", "",
        ".. code-block:: python", "",
        *["   " + line for line in pformat(columns, width=72).splitlines()], "",
        ".. raw:: html", "", "   </details>", "",
    ])


def setup(app):
    app.add_directive("model-reference", ModelReference)
    app.connect("autodoc-process-signature", compact_signature)
    app.connect("autodoc-process-docstring", document_defaults, priority=400)
    return {"version": "2", "parallel_read_safe": True, "parallel_write_safe": True}
