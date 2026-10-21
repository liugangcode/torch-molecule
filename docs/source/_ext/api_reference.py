"""Format API fields and long dataset defaults without changing the Python API."""

import inspect
from pprint import pformat

from docutils import nodes


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


def parameter_definitions(app, doctree, docname):
    """Give parameter names and types their own line, retaining nested content."""
    for field in doctree.findall(nodes.field):
        if field[0].astext() not in {"Parameters", "Keyword Arguments", "Other Parameters"}:
            continue
        body = field[1]
        items = list(body[0]) if len(body) == 1 and isinstance(body[0], nodes.bullet_list) else [body]
        definitions = nodes.definition_list(classes=["tm-parameters"])
        for item in items:
            if not len(item) or not isinstance(item[0], nodes.paragraph):
                break
            paragraph = item[0]
            separator = next((i for i, child in enumerate(paragraph)
                              if isinstance(child, nodes.Text) and str(child) == " – "), None)
            if separator is None:
                separator = len(paragraph)
            term = nodes.term("", "", *[child.deepcopy() for child in paragraph[:separator]])
            description = nodes.paragraph("", "", *[child.deepcopy() for child in paragraph[separator + 1:]])
            definition = nodes.definition("", description, *[child.deepcopy() for child in item[1:]])
            definitions += nodes.definition_list_item("", term, definition)
        else:
            body.children = []
            body += definitions


def setup(app):
    app.connect("autodoc-process-signature", compact_signature)
    app.connect("autodoc-process-docstring", document_defaults, priority=400)
    app.connect("doctree-resolved", parameter_definitions)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
