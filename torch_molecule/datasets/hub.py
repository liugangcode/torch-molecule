"""Load datasets from the molecular datasets collection on Hugging Face.

The collection publishes one repository per dataset and an index
(``liuganghuggingface/molecular-datasets-index``) that lists every dataset with
its license, access, original sources, quality tags, tables, splits, and
targets. Browse it at https://liugangcode.github.io/molecular-dataset-hub/.

Datasets whose license does not allow redistribution are listed with
``access="link"``; loading one raises an error that names the original source.
Gated datasets (noncommercial terms) need ``huggingface-cli login`` and an
accepted access request on the dataset page.

``Hub`` holds one index and the place its files come from; the module-level
functions use a default ``Hub``. Point a ``Hub`` at a mirror or a local copy with
``index_repo`` or ``index_path``.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
# Rows a split scheme leaves out.
UNASSIGNED = "unassigned"
SUMMARY_FIELDS = ["name", "title", "domain", "category", "application", "access", "license_class", "label_origin"]


@lru_cache(maxsize=8)
def _read_index(path: str, mtime: float) -> Tuple[dict, ...]:
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    if index["schema"] > SUPPORTED_SCHEMA:
        raise RuntimeError(f"The dataset index uses schema {index['schema']}; this torch-molecule reads "
                           f"schema {SUPPORTED_SCHEMA}. Upgrade torch-molecule, or pin an older release=.")
    return tuple(index["datasets"])


class Hub:
    """A release of the molecular datasets collection.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Download cache; files are fetched on first use.
    release : str, optional
        Collection release tag (e.g. ``"v2026.10.22"``) for reproducible results; default the
        latest release.
    index_repo : str, default="liuganghuggingface/molecular-datasets-index"
        Hugging Face dataset repository that holds ``index.json``.
    index_path : str, optional
        A local ``index.json`` whose parent folder sits next to one folder per dataset
        repository. Defaults to the ``TORCHMOL_DATASET_INDEX`` environment variable.
    """

    def __init__(self, local_dir: str = "torchmol_data", release: Optional[str] = None,
                 index_repo: str = INDEX_REPO, index_path: Optional[str] = None):
        self.local_dir = local_dir
        self.release = release
        self.index_repo = index_repo
        self.index_path = index_path or os.environ.get(INDEX_ENV)

    # ---- files ----------------------------------------------------------------------------

    def fetch(self, repo_id: str, filename: str, revision: Optional[str] = None) -> str:
        """Local path of one file of a collection repository, downloading it if needed."""
        if self.index_path:
            repos = os.path.dirname(os.path.dirname(os.path.abspath(self.index_path)))
            return (self.index_path if repo_id == self.index_repo and filename == "index.json"
                    else os.path.join(repos, repo_id.split("/")[-1], filename))
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision,
                               cache_dir=os.path.join(self.local_dir, "hub"))

    # ---- index ----------------------------------------------------------------------------

    def entries(self) -> List[dict]:
        """Every index entry of this release."""
        path = self.fetch(self.index_repo, "index.json", self.release)
        return list(_read_index(path, os.path.getmtime(path)))

    def info(self, name: str) -> dict:
        """Full index entry: license, links to original sources, quality tags, tables, and targets."""
        for entry in self.entries():
            if entry["name"] == name:
                return entry
        raise KeyError(f"Unknown dataset {name!r}; see list_datasets()")

    def list(self, search: Optional[str] = None, where: Optional[Callable[[dict], bool]] = None,
             **fields) -> List[dict]:
        """Summaries of datasets, filtered by text, by index fields, or by a predicate.

        ``fields`` match summary fields exactly (``category="adme"``, ``grade="ready"``,
        ``split="source"``); ``where`` receives the full index entry.
        """
        rows = []
        for entry in self.entries():
            row = _summary(entry)
            if any(value is not None and row.get(key) != value for key, value in fields.items()):
                continue
            text = " ".join(str(entry.get(k) or "") for k in ("name", "title", "description", "application"))
            if search and search.lower() not in text.lower():
                continue
            if where is not None and not where(entry):
                continue
            rows.append(row)
        return rows

    # ---- tables ---------------------------------------------------------------------------

    def _table(self, name: str, config: Optional[str]) -> Tuple[dict, dict]:
        entry = self.info(name)
        if entry["access"] == "link":
            source = entry["links"].get("original_data") or (entry["links"]["studies"] or [{}])[0].get("url")
            raise ValueError(f"{name} is not redistributed ({entry['access_reason']}). Get it from: {source}")
        if not entry["tables"]:
            files = [m["path"] for m in entry.get("modalities", []) if m["kind"] == "file"]
            raise ValueError(f"{name} has no SMILES table; get its files with download_files({name!r}): {files}")
        table = (entry["tables"][0] if config is None
                 else next((t for t in entry["tables"] if t["config"] == config), None))
        if table is None:
            raise KeyError(f"No table {config!r} in {name}; tables: {[t['config'] for t in entry['tables']]}")
        return entry, table

    def table(self, name: str, config: Optional[str] = None, split: Optional[str] = None,
              scheme: Optional[str] = None, columns: Optional[Sequence[str]] = None):
        """Rows of one published table as a ``pyarrow.Table``, with every source column.

        Use this for fields ``load_dataset`` does not turn into targets: text answers,
        conditions, identifiers, dates. ``split`` and ``scheme`` work as in ``load_dataset``;
        ``table.to_pandas()`` gives a DataFrame.
        """
        entry, table = self._table(name, config)
        return self._rows(entry, table, split, scheme, columns)

    def task_values(self, name: str, config: Optional[str] = None) -> List[str]:
        """Values of a long-format table's task column (assays, targets), most-labeled first."""
        from collections import Counter

        entry, table = self._table(name, config)
        column = table["task_column"]
        if not column:
            targets = [t["name"] for t in table["targets"]]
            raise ValueError(f"{name}/{table['config']} is not long-format; its targets are {targets}")
        counts = Counter(self._rows(entry, table, None, None, [column])[column].to_pylist())
        counts.pop(None, None)
        return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))]

    def observations(self, name: str, config: Optional[str] = None):
        """Every source row of a table before identical structures were merged, as a ``pyarrow.Table``."""
        import pyarrow.parquet as pq

        entry, table = self._table(name, config)
        return pq.read_table(self.fetch(entry["repo_id"], table["observations"], entry["revision"]))

    def _rows(self, entry: dict, table: dict, split: Optional[str], scheme: Optional[str],
              columns: Optional[Sequence[str]]):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        files, column = select_split(table, split, scheme)
        wanted = None if columns is None else list(dict.fromkeys([*columns, *([column] if column else [])]))
        parts = []
        for path in files:
            local = self.fetch(entry["repo_id"], path, entry["revision"])
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

    def parts(self, name: str, config: Optional[str] = None, scheme: Optional[str] = None) -> List[str]:
        """Part names (``"train"``, ``"test"``, ...) of a table's split; ``["full"]`` when the source gives none."""
        entry, table = self._table(name, config)
        return self._parts(entry, table, scheme)

    def _parts(self, entry: dict, table: dict, scheme: Optional[str]) -> List[str]:
        if scheme in (None, default_scheme(table)):
            select_split(table, None, scheme)
            return list(table["files"])
        column = "split_" + scheme
        values = set(self._rows(entry, table, None, scheme, [column])[column].to_pylist())
        order = {"train": 0, "validation": 1, "test": 2}
        return sorted(values, key=lambda part: (order.get(part, 3), part))

    # ---- datasets -------------------------------------------------------------------------

    def load(self, name: str, targets: Optional[Sequence[str]] = None, config: Optional[str] = None,
             split: Optional[str] = None, scheme: Optional[str] = None,
             task_values: Optional[Sequence[str]] = None, columns: Optional[Sequence[str]] = None,
             include_censored: bool = False, drop_leaked: bool = True) -> SMILESDataset:
        """See :func:`load_dataset`."""
        import pyarrow as pa

        entry, table = self._table(name, config)
        target_info = {t["name"]: t for t in table["targets"]}
        chosen = list(targets) if targets is not None else list(target_info)
        unknown = set(chosen) - set(target_info)
        if unknown:
            raise KeyError(f"Unknown targets {sorted(unknown)}; available: {list(target_info)}")
        if table["task_column"] and not task_values:
            raise ValueError(f"{name}/{table['config']} is long-format by {table['task_column']!r}; "
                             f"pass task_values=[...], for example from task_values({name!r})")
        extra = list(columns or [])
        needed = ["structure_key", *chosen, *([table["task_column"]] if table["task_column"] else []), *extra]
        rows = self._rows(entry, table, split, scheme, needed)
        missing = [c for c in extra if c not in rows.column_names]
        if missing:
            raise KeyError(f"No columns {missing} in {name}/{table['config']}; see load_table({name!r}).schema")
        parts = self._parts(entry, table, scheme)
        if split not in (None, "train") and drop_leaked and "train" in parts:
            seen = set(self._rows(entry, table, "train", scheme, ["structure_key"])["structure_key"].to_pylist())
            rows = rows.filter(pa.array([key not in seen for key in rows["structure_key"].to_pylist()]))
        values = rows.to_pydict()
        if table["task_column"]:
            index, target, target_names = _long_targets(values, table["task_column"], chosen, task_values,
                                                        include_censored)
        else:
            index = list(range(rows.num_rows))
            target = np.column_stack([_labels(values[t], include_censored) for t in chosen]) if chosen else None
            target_names = chosen
        keys = values["structure_key"]
        return SMILESDataset(
            data=[".".join(json.loads(keys[i])) for i in index],
            target=target,
            target_names=target_names,
            groups=[keys[i] for i in index],
            extra={c: [values[c][i] for i in index] for c in extra} or None,
            info={"name": name, "config": table["config"], "split": split, "scheme": scheme or default_scheme(table),
                  "splits": parts, "schemes": table.get("split_schemes", []),
                  "targets": {t: target_info[t] for t in chosen}, "label_origin": entry["label_origin"],
                  "license": entry["license"], "access": entry["access"], "links": entry["links"],
                  "quality": entry["quality"], "repo_id": entry["repo_id"], "revision": entry["revision"]},
        )

    def download(self, name: str, patterns: Optional[Sequence[str]] = None) -> List[str]:
        """Download a dataset's native files (structures, spectra, images, archives); returns local paths.

        ``patterns`` are shell-style globs over the paths in ``dataset_info(name)["modalities"]``.
        """
        entry = self.info(name)
        if entry["access"] == "link":
            self._table(name, None)  # raises with the original source
        files = [m["path"] for m in entry.get("modalities", []) if m["kind"] == "file"]
        chosen = [f for f in files if not patterns or any(fnmatch.fnmatch(f, p) for p in patterns)]
        if not chosen:
            raise KeyError(f"No files of {name} match {list(patterns or [])}; files: {files}")
        return [self.fetch(entry["repo_id"], path, entry["revision"]) for path in chosen]


# ---- splits ---------------------------------------------------------------------------------
# A table's "split" field says how the source divides it ("kind"). Every published table has one
# file per part of its default division; a source with several divisions ("split_schemes", e.g.
# WelQrate's five scaffold and five random splits) also carries one "split_<scheme>" column per
# division in every file.

def default_scheme(table: dict) -> Optional[str]:
    schemes = table.get("split_schemes") or []
    return schemes[0] if schemes else None


def select_split(table: dict, split: Optional[str], scheme: Optional[str]) -> Tuple[List[str], Optional[str]]:
    """Files to read and the column to filter on (``None``: take the files as they are)."""
    schemes = table.get("split_schemes") or []
    if scheme is not None and scheme not in schemes:
        raise KeyError(f"No split scheme {scheme!r} in {table['config']}; schemes: {schemes or 'none'}")
    files = table["files"]
    if scheme in (None, default_scheme(table)):
        if split is None:
            return list(files.values()), None
        if split not in files:
            raise KeyError(f"No split {split!r} in {table['config']}; splits: {list(files)}")
        return [files[split]], None
    return list(files.values()), "split_" + scheme


# ---- labels ---------------------------------------------------------------------------------

def _summary(entry: dict) -> dict:
    tables = entry["tables"]
    # A table without a source split publishes one file, "full".
    divided = any(set(t["files"]) - {"full"} for t in tables)
    return ({key: entry.get(key) for key in SUMMARY_FIELDS}
            | {"grade": entry["quality"]["grade"], "tables": len(tables),
               "split": "source" if divided else "none",
               "schemes": sorted({s for t in tables for s in t.get("split_schemes", [])})})


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


# ---- module-level interface -------------------------------------------------------------------

def list_datasets(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    grade: Optional[str] = None,
    access: Optional[str] = None,
    license_class: Optional[str] = None,
    split: Optional[str] = None,
    search: Optional[str] = None,
    where: Optional[Callable[[dict], bool]] = None,
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
    split : {"source", "none"}, optional
        ``"source"``: the source divides the data into training and test sets.
    search : str, optional
        Text to find in the name, title, description, or application.
    where : callable, optional
        Keeps a dataset when ``where(dataset_info(name))`` is true.
    release : str, optional
        Collection release tag (e.g. ``"v2026.10.22"``); default the latest release.

    Returns
    -------
    list of dict
        One summary per dataset, with ``grade``, ``tables``, ``split`` and the names of its
        split ``schemes``.
    """
    return Hub(local_dir, release).list(search=search, where=where, domain=domain, category=category, grade=grade,
                                        access=access, license_class=license_class, split=split)


def dataset_info(name: str, local_dir: str = "torchmol_data", release: Optional[str] = None) -> dict:
    """Full index entry: license, links to original sources, quality tags, tables, splits, and targets."""
    return Hub(local_dir, release).info(name)


def load_dataset(
    name: str,
    targets: Optional[Sequence[str]] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    scheme: Optional[str] = None,
    task_values: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
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
        A source split (``"train"``, ``"validation"``, ``"test"``, ...). Default: all rows.
        ``dataset_info(name)["tables"][i]["files"]`` lists the parts.
    scheme : str, optional
        For sources with several splits (``table["split_schemes"]``, e.g. ``"scaffold_3"``
        or ``"random_1"`` in WelQrate), which one ``split`` refers to. Default the first.
        Rows a scheme leaves out are not loaded.
    task_values : list of str, optional
        For long-format tables (one row per measurement, with a task column such as an
        assay ID), the task values to load; each becomes one target column, and rows
        keep NaN for the other tasks. Required for long-format tables.
    columns : list of str, optional
        Other source columns to return in ``dataset.extra`` (conditions, identifiers,
        dates, text). ``load_table(name).schema`` lists them.
    include_censored : bool, default=False
        Censored labels such as ``<10`` become NaN unless True, in which case the
        bound (10) is used.
    drop_leaked : bool, default=True
        When loading a non-train split, drop molecules that also appear in the
        source's train split (of the same scheme).
    release : str, optional
        Collection release tag (e.g. ``"v2026.10.22"``) for reproducible results; default the
        latest release. New datasets appear here as soon as the collection publishes them.

    Returns
    -------
    SMILESDataset
        ``data`` holds canonical SMILES (mixture components joined with ``.``),
        ``target`` is (rows, targets) with NaN for missing labels, ``groups`` holds the
        structure key of each row (use it to keep one molecule in one split), ``extra``
        the requested ``columns``, and ``info`` units, label kinds, splits, license,
        original sources, and quality tags.
    """
    return Hub(local_dir, release).load(name, targets=targets, config=config, split=split, scheme=scheme,
                                        task_values=task_values, columns=columns,
                                        include_censored=include_censored, drop_leaked=drop_leaked)


def load_table(name: str, config: Optional[str] = None, split: Optional[str] = None, scheme: Optional[str] = None,
               columns: Optional[Sequence[str]] = None, local_dir: str = "torchmol_data",
               release: Optional[str] = None):
    """Rows of one published table as a ``pyarrow.Table`` with every source column.

    For data that is not a SMILES-to-number task: question and answer text, reaction or
    mixture records, conditions. ``split`` and ``scheme`` work as in :func:`load_dataset`.
    Call ``.to_pandas()`` on the result for a DataFrame.
    """
    return Hub(local_dir, release).table(name, config=config, split=split, scheme=scheme, columns=columns)


def task_values(name: str, config: Optional[str] = None, local_dir: str = "torchmol_data",
                release: Optional[str] = None) -> List[str]:
    """Task values (assay or target IDs) of a long-format table, most-labeled first.

    Pass some of them to ``load_dataset(name, task_values=[...])``; each becomes one target column.
    """
    return Hub(local_dir, release).task_values(name, config)


def download_files(name: str, patterns: Optional[Sequence[str]] = None, local_dir: str = "torchmol_data",
                   release: Optional[str] = None) -> List[str]:
    """Download a dataset's native files (3D structures, spectra, images, archives).

    Parameters
    ----------
    patterns : list of str, optional
        Shell-style patterns such as ``["structures/*.zip"]``; default every file listed in
        ``dataset_info(name)["modalities"]``.

    Returns
    -------
    list of str
        Local paths, in the order of the index.
    """
    return Hub(local_dir, release).download(name, patterns)


def _index(local_dir: str, release: Optional[str] = None) -> List[dict]:
    """Index entries of a release; kept for callers of the previous interface."""
    return Hub(local_dir, release).entries()
