"""Load datasets from the molecular dataset hub on Hugging Face.

Each dataset is one Hugging Face repository; an index
(``liuganghuggingface/molecular-datasets-index``) lists every dataset with its license,
original sources, quality grade, tables, splits, and targets. Browse it at
https://liugangcode.github.io/molecular-dataset-hub/.

A dataset can hold several tables (configs): MoleculeACE has one per protein target, and
``load_dataset("moleculeace", "chembl204_ki")`` loads one of them.

Datasets whose license does not allow redistribution are listed with ``access="link"``;
loading one raises an error that names the original source. Gated datasets need
``huggingface-cli login`` and an accepted access request on the dataset page.

Files are cached in ``~/torchmol_data``; set ``TORCHMOL_DATA`` or pass ``local_dir`` to
use another folder. ``TORCHMOL_DATASET_INDEX`` reads a local copy of the index instead:
the path of an ``index.json`` whose parent folder sits next to one folder per dataset
repository (e.g. ``moldata-aqsoldb``).
"""
from __future__ import annotations

from collections import Counter
import fnmatch
import json
import os
import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .constant import SMILESDataset

INDEX_REPO = "liuganghuggingface/molecular-datasets-index"
# Index format this loader understands; the hub bumps it when a field changes meaning.
SUPPORTED_SCHEMA = 1
MISSING = {"", "nan", "na", "n/a", "null", "none"}
CENSORED = re.compile(r"^\s*[<>~]=?\s*")
UNASSIGNED = "unassigned"  # rows a split scheme leaves out


# ---- files and index ------------------------------------------------------------------------

def _home(local_dir: Optional[str]) -> str:
    return os.path.expanduser(local_dir or os.environ.get("TORCHMOL_DATA", "~/torchmol_data"))


def _fetch(repo_id: str, filename: str, revision: Optional[str], local_dir: Optional[str]) -> str:
    """Local path of one file of a hub repository, downloading it on first use."""
    index = os.environ.get("TORCHMOL_DATASET_INDEX")
    if index:
        if repo_id == INDEX_REPO and filename == "index.json":
            return index
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(index))), repo_id.split("/")[-1], filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision,
                           cache_dir=os.path.join(_home(local_dir), "hub"))


@lru_cache(maxsize=8)
def _read_index(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    if index["schema"] > SUPPORTED_SCHEMA:
        raise RuntimeError(f"The dataset index uses schema {index['schema']}; this torch-molecule reads "
                           f"schema {SUPPORTED_SCHEMA}. Upgrade torch-molecule, or pin an older release=.")
    return index


def _index(local_dir: Optional[str] = None, release: Optional[str] = None) -> dict:
    path = _fetch(INDEX_REPO, "index.json", release, local_dir)
    return _read_index(path, os.path.getmtime(path))


def _entry(name: str, config: Optional[str], local_dir, release) -> Tuple[dict, Optional[str]]:
    """Index entry of a dataset and the config asked for; former names resolve to their new place."""
    index = _index(local_dir, release)
    moved = index.get("renamed", {}).get(name)
    if moved:
        name, config = moved["name"], config or moved.get("config")
    for entry in index["datasets"]:
        if entry["name"] == name:
            return entry, config
    raise KeyError(f"Unknown dataset {name!r}; see list_datasets()")


def _table(name: str, config: Optional[str], local_dir, release) -> Tuple[dict, dict]:
    entry, config = _entry(name, config, local_dir, release)
    if entry["access"] == "link":
        source = entry["links"].get("original_data") or (entry["links"]["studies"] or [{}])[0].get("url")
        raise ValueError(f"{name} is not redistributed ({entry['access_reason']}). Get it from: {source}")
    tables = {table["config"]: table for table in entry["tables"]}
    if not tables:
        files = [m["path"] for m in entry.get("modalities", []) if m["kind"] == "file"]
        raise ValueError(f"{name} has no SMILES table; get its files with download_files({name!r}): {files}")
    # Releases before configs were named load their first table by default.
    config = config or entry.get("default_config", entry["tables"][0]["config"])
    if config is None:
        raise ValueError(f"{entry['name']} has {len(tables)} configs; name one, for example "
                         f"load_dataset({entry['name']!r}, {next(iter(tables))!r}). Configs: {list(tables)}")
    if config not in tables:
        raise KeyError(f"No config {config!r} in {entry['name']}; configs: {list(tables)}")
    return entry, tables[config]


# ---- splits ---------------------------------------------------------------------------------
# Every published table has one file per part of its source split ("full" when there is none).
# A source with several splits ("split_schemes", e.g. WelQrate's five scaffold and five random
# splits) also carries one "split_<scheme>" column per split in every file; files follow the first.

def _select(table: dict, split: Optional[str], scheme: Optional[str]) -> Tuple[List[str], Optional[str]]:
    """Files to read and the column to filter on (``None``: take the files as they are)."""
    schemes = table.get("split_schemes") or []
    if scheme is not None and scheme not in schemes:
        raise KeyError(f"No split scheme {scheme!r} in {table['config']}; schemes: {schemes or 'none'}")
    files = table["files"]
    if scheme in (None, *schemes[:1]):
        if split is None:
            return list(files.values()), None
        if split not in files:
            raise KeyError(f"No split {split!r} in {table['config']}; splits: {list(files)}")
        return [files[split]], None
    return list(files.values()), "split_" + scheme


def _rows(entry: dict, table: dict, split, scheme, columns, local_dir):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files, column = _select(table, split, scheme)
    wanted = None if columns is None else list(dict.fromkeys([*columns, *([column] if column else [])]))
    parts = []
    for path in files:
        local = _fetch(entry["repo_id"], path, entry["revision"], local_dir)
        present = None if wanted is None else [c for c in wanted if c in pq.read_schema(local).names]
        parts.append(pq.read_table(local, columns=present))
    rows = pa.concat_tables(parts, promote_options="default") if len(parts) > 1 else parts[0]
    if column:
        kept = rows.filter(pc.equal(rows[column], split) if split else pc.not_equal(rows[column], UNASSIGNED))
        if split and not kept.num_rows:
            found = sorted(set(rows[column].to_pylist()) - {UNASSIGNED})
            raise KeyError(f"No split {split!r} in scheme {scheme!r} of {table['config']}; splits: {found}")
        rows = kept
    return rows


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
    wanted = set(task_values)
    rows = [i for i, value in enumerate(columns[task_column]) if value in wanted]
    names = [f"{t}/{v}" if len(targets) > 1 else str(v) for t in targets for v in task_values]
    out = np.full((len(rows), len(names)), np.nan)
    for j, (t, v) in enumerate((t, v) for t in targets for v in task_values):
        labels = _labels([columns[t][i] for i in rows], include_censored)
        mask = np.array([columns[task_column][i] == v for i in rows], dtype=bool)
        out[mask, j] = labels[mask]
    return rows, out, names


# ---- public interface -----------------------------------------------------------------------

def load_dataset(
    name: str,
    config: Optional[str] = None,
    *,
    split: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
    return_X_y: bool = False,
    scheme: Optional[str] = None,
    task_values: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
    include_censored: bool = False,
    drop_leaked: bool = True,
    local_dir: Optional[str] = None,
    release: Optional[str] = None,
):
    """Load a hub dataset: SMILES and their labels.

    Parameters
    ----------
    name : str
        Dataset name from ``list_datasets()``, for example ``"qm9"`` or ``"moleculeace"``.
    config : str, optional
        Which table, for datasets with several (one per target, assay, or property);
        ``dataset_info(name)["tables"]`` lists them. Datasets with one table need none.
    split : str, optional
        A source split such as ``"train"`` or ``"test"``; default all rows.
    targets : list of str, optional
        Label columns to load; default all of the table's.
    return_X_y : bool, default=False
        Return ``(smiles, target)`` instead of a ``SMILESDataset``.
    scheme : str, optional
        For sources with several splits (``"scaffold_3"``, ``"random_1"`` in WelQrate),
        which one ``split`` refers to; default the first. Rows it leaves out are not loaded.
    task_values : list of str, optional
        For tables with one row per measurement and a task column (an assay or target ID):
        the tasks to load, each as one target column. :func:`task_values` lists them.
    columns : list of str, optional
        Other source columns to return in ``dataset.extra`` (identifiers, conditions, dates).
    include_censored : bool, default=False
        Censored labels such as ``<10`` load as NaN, or as their bound (10) when True.
    drop_leaked : bool, default=True
        A non-train split leaves out molecules that are also in the train split.
    local_dir : str, optional
        Download cache; default ``$TORCHMOL_DATA`` or ``~/torchmol_data``.
    release : str, optional
        Hub release tag (e.g. ``"v2026.10.31.2"``) for reproducible results; default the latest.

    Returns
    -------
    SMILESDataset or tuple
        ``data`` holds canonical SMILES (mixture components joined with ``.``), ``target``
        an (n_molecules, n_targets) array with NaN for missing labels, ``groups`` the
        structure key of each row, ``extra`` the requested ``columns``, and ``info`` the
        description, units, readable label names, splits, license, sources, and quality.
        With ``return_X_y=True``, ``(data, target)``.
    """
    import pyarrow as pa

    entry, table = _table(name, config, local_dir, release)
    target_info = {t["name"]: t for t in table["targets"]}
    chosen = list(target_info) if targets is None else list(targets)
    unknown = set(chosen) - set(target_info)
    if unknown:
        raise KeyError(f"Unknown targets {sorted(unknown)}; available: {list(target_info)}")
    task_column = table["task_column"]
    if task_column and not task_values:
        raise ValueError(f"{entry['name']}/{table['config']} has one row per measurement, by {task_column!r}; "
                         f"pass task_values=[...], for example from task_values({name!r})")
    extra = list(columns or [])
    rows = _rows(entry, table, split, scheme, ["structure_key", *chosen, *([task_column] if task_column else []), *extra],
                 local_dir)
    missing = [c for c in extra if c not in rows.column_names]
    if missing:
        raise KeyError(f"No columns {missing} in {entry['name']}/{table['config']}; see load_table({name!r}).schema")
    if split not in (None, "train") and drop_leaked and "train" in table["files"]:
        seen = set(_rows(entry, table, "train", scheme, ["structure_key"], local_dir)["structure_key"].to_pylist())
        rows = rows.filter(pa.array([key not in seen for key in rows["structure_key"].to_pylist()]))
    values = rows.to_pydict()
    if task_column:
        index, target, target_names = _long_targets(values, task_column, chosen, task_values, include_censored)
    else:
        index = list(range(rows.num_rows))
        target = np.column_stack([_labels(values[t], include_censored) for t in chosen]) if chosen else None
        target_names = chosen
    smiles = [".".join(json.loads(values["structure_key"][i])) for i in index]
    if return_X_y:
        return smiles, target
    return SMILESDataset(
        data=smiles,
        target=target,
        target_names=target_names,
        groups=[values["structure_key"][i] for i in index],
        extra={c: [values[c][i] for i in index] for c in extra} or None,
        info={"name": entry["name"], "config": table["config"], "title": table.get("title", entry["title"]),
              "description": entry.get("description"), "split": split, "splits": list(table["files"]),
              "scheme": scheme or next(iter(table.get("split_schemes") or []), None), "schemes": table.get("split_schemes", []),
              "targets": {t: target_info[t] for t in chosen}, "label_origin": entry["label_origin"],
              "license": entry["license"], "access": entry["access"], "links": entry["links"],
              "quality": table.get("quality", entry["quality"]), "repo_id": entry["repo_id"],
              "revision": entry["revision"]},
    )


def list_datasets(
    search: Optional[str] = None,
    *,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    grade: Optional[str] = None,
    access: Optional[str] = None,
    license_class: Optional[str] = None,
    local_dir: Optional[str] = None,
    release: Optional[str] = None,
) -> List[dict]:
    """Datasets in the hub, optionally filtered.

    Parameters
    ----------
    search : str, optional
        Text to find in a dataset's name, title, description, config titles, or label
        names, such as ``"thrombin"`` or ``"solubility"``.
    domain : {"small_molecules", "polymers"}, optional
    category : str, optional
        For example ``"adme"``, ``"toxicity"``, ``"bioactivity"``.
    grade : {"ready", "care", "caution", "unmeasured"}, optional
        Data-quality grade of the dataset or of any of its configs.
    access : {"public", "gated", "link"}, optional
        ``link`` datasets are not redistributed; get them from their original source.
    license_class : {"open", "share_alike", "noncommercial", "unverified"}, optional

    Returns
    -------
    list of dict
        One summary per dataset: name, title, domain, category, grade, access,
        license class, and its configs.
    """
    rows = []
    for entry in _index(local_dir, release)["datasets"]:
        configs = [t["config"] for t in entry["tables"]]
        grades = {entry["quality"]["grade"]} | {t["quality"]["grade"] for t in entry["tables"] if "quality" in t}
        text = " ".join([entry["name"], entry["title"], entry.get("description") or "", entry["application"],
                         *(t.get("title", "") for t in entry["tables"]),
                         *(target.get("label", "") for t in entry["tables"] for target in t["targets"])])
        if (search and search.lower() not in text.lower()) or (grade and grade not in grades) or any(
                value and entry[key] != value for key, value in
                [("domain", domain), ("category", category), ("access", access), ("license_class", license_class)]):
            continue
        rows.append({"name": entry["name"], "title": entry["title"], "domain": entry["domain"],
                     "category": entry["category"], "grade": entry["quality"]["grade"], "access": entry["access"],
                     "license_class": entry["license_class"], "configs": configs})
    return rows


def dataset_info(name: str, *, local_dir: Optional[str] = None, release: Optional[str] = None) -> dict:
    """Everything the hub knows about a dataset: description, license, original sources,
    quality, and per config its title, splits, and targets with units and readable names."""
    return _entry(name, None, local_dir, release)[0]


def task_values(name: str, config: Optional[str] = None, *, local_dir: Optional[str] = None,
                release: Optional[str] = None) -> List[str]:
    """Task values (assay or target IDs) of a table with one row per measurement, most-measured first.

    Pass some to ``load_dataset(name, task_values=[...])``; each becomes one target column.
    """
    entry, table = _table(name, config, local_dir, release)
    column = table["task_column"]
    if not column:
        raise ValueError(f"{entry['name']}/{table['config']} has one column per target: "
                         f"{[t['name'] for t in table['targets']]}")
    counts = Counter(_rows(entry, table, None, None, [column], local_dir)[column].to_pylist())
    counts.pop(None, None)
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))]


def load_table(name: str, config: Optional[str] = None, *, split: Optional[str] = None,
               scheme: Optional[str] = None, columns: Optional[Sequence[str]] = None,
               local_dir: Optional[str] = None, release: Optional[str] = None):
    """Every column of a table as a ``pyarrow.Table`` (``.to_pandas()`` for a DataFrame).

    For data that is not SMILES and numbers: question and answer text, reactions, mixtures,
    conditions. ``split`` and ``scheme`` work as in :func:`load_dataset`.
    """
    entry, table = _table(name, config, local_dir, release)
    return _rows(entry, table, split, scheme, columns, local_dir)


def download_files(name: str, patterns: Optional[Sequence[str]] = None, *, local_dir: Optional[str] = None,
                   release: Optional[str] = None) -> List[str]:
    """Download a dataset's native files (3D structures, spectra, images, archives).

    Parameters
    ----------
    patterns : list of str, optional
        Shell-style patterns such as ``["structures/*"]``; default every file listed in
        ``dataset_info(name)["modalities"]``.

    Returns
    -------
    list of str
        Local paths.
    """
    entry, _ = _entry(name, None, local_dir, release)
    if entry["access"] == "link":
        _table(name, None, local_dir, release)  # raises with the original source
    files = [m["path"] for m in entry.get("modalities", []) if m["kind"] == "file"]
    chosen = [f for f in files if not patterns or any(fnmatch.fnmatch(f, p) for p in patterns)]
    if not chosen:
        raise KeyError(f"No files of {name} match {list(patterns or [])}; files: {files}")
    return [_fetch(entry["repo_id"], path, entry["revision"], local_dir) for path in chosen]
