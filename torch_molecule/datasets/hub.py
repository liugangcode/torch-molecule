"""Load datasets from the molecular datasets collection on Hugging Face.

The collection publishes one repository per dataset and an index
(``liuganghuggingface/molecular-datasets-index``) that lists every dataset with
its license, access, original sources, quality tags, tables, splits, and
targets. Browse it at https://liugangcode.github.io/molecular-dataset-hub/.

Datasets whose license does not allow redistribution are listed with
``access="link"``; loading one raises an error that names the original source.
Gated datasets (noncommercial terms) need ``huggingface-cli login`` and an
accepted access request on the dataset page.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constant import SMILESDataset

INDEX_REPO = "liuganghuggingface/molecular-datasets-index"
# Index format this loader understands; the collection bumps it when a field changes meaning.
SUPPORTED_SCHEMA = 1
# Read a local copy instead of the Hub: the path of an index.json whose parent folder sits
# next to one folder per dataset repo, named like the repo (e.g. "moldata-aqsoldb").
INDEX_ENV = "TORCHMOL_DATASET_INDEX"
MISSING = {"", "nan", "na", "n/a", "null", "none"}
CENSORED = re.compile(r"^\s*[<>~]=?\s*")


def _download(repo_id: str, filename: str, revision: Optional[str], cache_dir: str) -> str:
    local = os.environ.get(INDEX_ENV)
    if local:
        repos = os.path.dirname(os.path.dirname(os.path.abspath(local)))
        return local if filename == "index.json" else os.path.join(repos, repo_id.split("/")[-1], filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)


def _index(local_dir: str, release: Optional[str] = None) -> List[dict]:
    """Index entries of a collection release (a tag such as "v2026.10.22"), or the latest one."""
    path = _download(INDEX_REPO, "index.json", release, os.path.join(local_dir, "hub"))
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    if index["schema"] > SUPPORTED_SCHEMA:
        raise RuntimeError(f"The dataset index uses schema {index['schema']}; this torch-molecule reads "
                           f"schema {SUPPORTED_SCHEMA}. Upgrade torch-molecule, or pin an older release=.")
    return index["datasets"]


def list_datasets(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    grade: Optional[str] = None,
    access: Optional[str] = None,
    license_class: Optional[str] = None,
    local_dir: str = "torchmol_data",
    release: Optional[str] = None,
) -> List[dict]:
    """Summaries of datasets in the collection, filtered by any given field.

    Parameters
    ----------
    domain : {"small_molecules", "polymers"}, optional
    category : str, optional
        For example ``"adme"``, ``"toxicity"``, ``"thermodynamics"``.
    grade : {"ready", "care", "caution", "unmeasured"}, optional
        Rough data-quality grade; see ``dataset_info(name)["quality"]`` for its tags.
    access : {"public", "gated", "link"}, optional
        ``link`` datasets are not redistributed; get them from their original source.
    license_class : {"open", "share_alike", "noncommercial", "unverified"}, optional
    release : str, optional
        Collection release tag (e.g. ``"v2026.10.22"``); default the latest release.
    """
    wanted = {"domain": domain, "category": category, "access": access, "license_class": license_class}
    rows = []
    for entry in _index(local_dir, release):
        if any(value is not None and entry[key] != value for key, value in wanted.items()):
            continue
        if grade is not None and entry["quality"]["grade"] != grade:
            continue
        rows.append({key: entry[key] for key in ["name", "title", "domain", "category", "application", "access",
                                                 "license_class", "label_origin"]}
                    | {"grade": entry["quality"]["grade"], "tables": len(entry["tables"])})
    return rows


def dataset_info(name: str, local_dir: str = "torchmol_data", release: Optional[str] = None) -> dict:
    """Full index entry: license, links to original sources, quality tags, tables, and targets."""
    for entry in _index(local_dir, release):
        if entry["name"] == name:
            return entry
    raise KeyError(f"Unknown dataset {name!r}; see list_datasets()")


def load_dataset(
    name: str,
    targets: Optional[Sequence[str]] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    task_values: Optional[Sequence[str]] = None,
    include_censored: bool = False,
    drop_leaked: bool = True,
    local_dir: str = "torchmol_data",
    release: Optional[str] = None,
) -> SMILESDataset:
    """Load one table of a collection dataset as a ``SMILESDataset``.

    Parameters
    ----------
    name : str
        Dataset name from ``list_datasets()``, for example ``"moleculeace_chembl204_ki"``.
    targets : list of str, optional
        Target columns; default all targets of the table.
    config : str, optional
        Table name; default the dataset's first table.
    split : str, optional
        A source split (``"train"``, ``"test"``, ...). Default: all rows.
    task_values : list of str, optional
        For long-format tables (one row per measurement, with a task column such as an
        assay ID), the task values to load; each becomes one target column, and rows
        keep NaN for the other tasks. Required for long-format tables.
    include_censored : bool, default=False
        Censored labels such as ``<10`` become NaN unless True, in which case the
        bound (10) is used.
    drop_leaked : bool, default=True
        When loading a non-train split, drop molecules that also appear in the
        source's train split.
    release : str, optional
        Collection release tag (e.g. ``"v2026.10.22"``) for reproducible results; default the
        latest release. New datasets appear here as soon as the collection publishes them.

    Returns
    -------
    SMILESDataset
        ``data`` holds canonical SMILES (mixture components joined with ``.``),
        ``target`` is (rows, targets) with NaN for missing labels, ``groups`` holds the
        structure key of each row (use it to keep one molecule in one split), and
        ``info`` holds units, label kinds, license, original sources, and quality tags.
    """
    entry = dataset_info(name, local_dir, release)
    if entry["access"] == "link":
        source = entry["links"].get("original_data") or (entry["links"]["studies"] or [{}])[0].get("url")
        raise ValueError(f"{name} is not redistributed ({entry['access_reason']}). Get it from: {source}")
    table = entry["tables"][0] if config is None else next((t for t in entry["tables"] if t["config"] == config), None)
    if table is None:
        raise KeyError(f"No table {config!r} in {name}; tables: {[t['config'] for t in entry['tables']]}")
    splits = list(table["files"])
    names = [split] if split is not None else splits
    if split is not None and split not in splits:
        raise KeyError(f"No split {split!r} in {name}/{table['config']}; splits: {splits}")

    columns = _read(entry, [table["files"][s] for s in names], local_dir)
    if split not in (None, "train") and drop_leaked and "train" in splits:
        seen = set(_read(entry, [table["files"]["train"]], local_dir)["structure_key"])
        keep = [key not in seen for key in columns["structure_key"]]
        columns = {column: [v for v, k in zip(values, keep) if k] for column, values in columns.items()}

    target_info = {t["name"]: t for t in table["targets"]}
    chosen = list(targets) if targets is not None else list(target_info)
    unknown = set(chosen) - set(target_info)
    if unknown:
        raise KeyError(f"Unknown targets {sorted(unknown)}; available: {list(target_info)}")
    if table["task_column"]:
        if not task_values:
            raise ValueError(f"{name}/{table['config']} is long-format by {table['task_column']!r}; pass task_values=[...]")
        rows, target, target_names = _long_targets(columns, table["task_column"], chosen, task_values, include_censored)
    else:
        rows = list(range(len(columns["structure_key"])))
        target = np.column_stack([_labels(columns[t], include_censored) for t in chosen]) if chosen else None
        target_names = chosen
    return SMILESDataset(
        data=[".".join(json.loads(columns["structure_key"][i])) for i in rows],
        target=target,
        target_names=target_names,
        groups=[columns["structure_key"][i] for i in rows],
        info={"name": name, "config": table["config"], "splits": names,
              "targets": {t: target_info[t] for t in chosen}, "label_origin": entry["label_origin"],
              "license": entry["license"], "access": entry["access"], "links": entry["links"],
              "quality": entry["quality"], "repo_id": entry["repo_id"], "revision": entry["revision"]},
    )


def _read(entry: dict, files: List[str], local_dir: str) -> Dict[str, list]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(_download(entry["repo_id"], f, entry["revision"], os.path.join(local_dir, "hub")))
              for f in files]
    return pa.concat_tables(tables, promote_options="default").to_pydict() if len(tables) > 1 else tables[0].to_pydict()


def _labels(values: Sequence[str], include_censored: bool) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, raw in enumerate(values):
        text = "" if raw is None else str(raw).strip()
        if text.lower() in MISSING:
            continue
        if CENSORED.match(text):
            if not include_censored:
                continue
            text = CENSORED.sub("", text)
        try:
            out[i] = float(text)
        except ValueError:
            pass
    return out


def _long_targets(columns, task_column, targets, task_values, include_censored):
    """Rows of the requested tasks; one column per (target, task value), each row filling only its own."""
    rows = [i for i, value in enumerate(columns[task_column]) if value in set(task_values)]
    names = [f"{t}/{v}" if len(targets) > 1 else str(v) for t in targets for v in task_values]
    out = np.full((len(rows), len(names)), np.nan)
    for j, (t, v) in enumerate((t, v) for t in targets for v in task_values):
        labels = _labels([columns[t][i] for i in rows], include_censored)
        mask = np.array([columns[task_column][i] == v for i in rows])
        out[mask, j] = labels[mask]
    return rows, out, names
