"""Build documentation from this checkout's public API and package metadata."""

import ast
from datetime import date
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

project = "torch-molecule"
author = "Gang Liu"
copyright = f"2025–{date.today().year}, {author}"
release = metadata["version"]
version = release
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "furo"
html_title = "torch-molecule"
html_static_path = ["_static"]
html_css_files = ["torch-molecule.css"]
html_js_files = ["torch-molecule.js"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "top_of_page_buttons": [],
    "light_css_variables": {
        "color-brand-primary": "#984b35",
        "color-brand-content": "#984b35",
        "color-background-primary": "#faf9f6",
        "color-background-secondary": "#f1f0eb",
        "color-foreground-primary": "#242721",
        "color-foreground-secondary": "#555b51",
        "color-foreground-muted": "#64685e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#e8a38b",
        "color-brand-content": "#e8a38b",
        "color-background-primary": "#20231f",
        "color-background-secondary": "#292d27",
        "color-foreground-primary": "#f5f3eb",
        "color-foreground-secondary": "#c1c5ba",
        "color-foreground-muted": "#c1c5ba",
    },
}
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}

exports = ast.parse((ROOT / "torch_molecule" / "__init__.py").read_text())
model_names = next(
    ast.literal_eval(node.value)
    for node in exports.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
)
html_context = {
    "model_counts": {
        name: sum(model.endswith(f"Molecular{suffix}") for model in model_names)
        for name, suffix in [
            ("predictors", "Predictor"),
            ("encoders", "Encoder"),
            ("generators", "Generator"),
        ]
    },
    "model_total": len(model_names),
}
