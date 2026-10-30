import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from torch_molecule.datasets import Hub, dataset_info, download_files, list_datasets, load_dataset, load_table


def _key(smiles):
    return json.dumps(smiles if isinstance(smiles, list) else [smiles], separators=(",", ":"))


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Two published datasets (one wide with splits, one long-format) and one link-only entry."""
    _write(tmp_path / "moldata-sol/data/t/train.parquet", [
        {"structure_key": _key("CCO"), "y": "1.5", "z": ""},
        {"structure_key": _key("CCN"), "y": "<2", "z": "0"},
    ])
    _write(tmp_path / "moldata-sol/data/t/test.parquet", [
        {"structure_key": _key("CCO"), "y": "1.4", "z": "1"},
        {"structure_key": _key(["CCC", "[Na+]"]), "y": "NA", "z": "1"},
    ])
    _write(tmp_path / "moldata-assays/data/t/full.parquet", [
        {"structure_key": _key("CCO"), "assay": "A1", "pchembl": "6.0"},
        {"structure_key": _key("CCN"), "assay": "A2", "pchembl": "7.0"},
        {"structure_key": _key("CCC"), "assay": "A3", "pchembl": "5.0"},
    ])
    # Two split schemes: the files follow "scaffold_1"; "random_1" is a column in every file.
    _write(tmp_path / "moldata-screen/data/t/train.parquet", [
        {"structure_key": _key("C1CC1"), "active": "1", "year": 2001, "split_scaffold_1": "train", "split_random_1": "test"},
        {"structure_key": _key("C1CCC1"), "active": "0", "year": 2005, "split_scaffold_1": "train", "split_random_1": "train"},
    ])
    _write(tmp_path / "moldata-screen/data/t/test.parquet", [
        {"structure_key": _key("c1ccccc1"), "active": "0", "year": 2010, "split_scaffold_1": "test", "split_random_1": "train"},
        {"structure_key": _key("C1CCCC1"), "active": "1", "year": 2012, "split_scaffold_1": "test", "split_random_1": "unassigned"},
    ])
    _write(tmp_path / "moldata-screen/observations/t.parquet", [{"structure_key": _key("C1CC1"), "active": "1"}] * 3)
    _write(tmp_path / "moldata-raw/files/qm7.mat", [])
    table = lambda files, targets, task_column=None: {  # noqa: E731
        "config": "t", "files": files, "task_column": task_column,
        "targets": [{"name": t, "kind": "regression", "units": "u"} for t in targets]}
    common = {"label_origin": "experiment", "license": "CC0-1.0", "license_class": "open", "revision": None,
              "quality": {"grade": "ready", "tags": []}, "category": "physicochemical", "application": "physicochemical/solubility",
              "domain": "small_molecules", "title": "T",
              "links": {"original_data": "https://example.org/data", "studies": [], "redistributor": None, "built_on": None}}
    index = {"schema": 1, "datasets": [
        {**common, "name": "sol", "access": "public", "repo_id": "u/moldata-sol",
         "tables": [table({"train": "data/t/train.parquet", "test": "data/t/test.parquet"}, ["y", "z"])]},
        {**common, "name": "assays", "access": "public", "repo_id": "u/moldata-assays", "category": "bioactivity",
         "tables": [table({"full": "data/t/full.parquet"}, ["pchembl"], task_column="assay")]},
        {**common, "name": "screen", "access": "public", "repo_id": "u/moldata-screen", "category": "bioactivity",
         "description": "virtual screening benchmark",
         "tables": [{**table({"train": "data/t/train.parquet", "test": "data/t/test.parquet"}, ["active"]),
                     "observations": "observations/t.parquet", "split_schemes": ["scaffold_1", "random_1"],
                     "split": {"kind": "scheme", "column": "split_scaffold_1", "map": {"train": "train", "test": "test"}}}]},
        {**common, "name": "closed", "access": "link", "access_reason": "license not yet verified", "repo_id": None,
         "license_class": "unverified", "tables": []},
        {**common, "name": "raw", "access": "public", "repo_id": "u/moldata-raw", "tables": [],
         "modalities": [{"kind": "file", "modality": "source files", "path": "files/qm7.mat"}]},
    ]}
    path = tmp_path / "molecular-datasets-index" / "index.json"
    path.parent.mkdir()
    path.write_text(json.dumps(index))
    monkeypatch.setenv("TORCHMOL_DATASET_INDEX", str(path))
    return tmp_path


def test_list_and_info(hub):
    assert [d["name"] for d in list_datasets(category="bioactivity")] == ["assays", "screen"]
    assert [d["name"] for d in list_datasets(access="link")] == ["closed"]
    assert dataset_info("sol")["license"] == "CC0-1.0"
    with pytest.raises(KeyError):
        dataset_info("absent")


def test_load_split_drops_leaked_molecules_and_censored_labels(hub):
    train = load_dataset("sol", split="train")
    assert train.data == ["CCO", "CCN"]
    assert np.isnan(train.target[1, 0])  # "<2" is censored
    assert load_dataset("sol", split="train", include_censored=True).target[1, 0] == 2.0
    test = load_dataset("sol", split="test")
    assert test.data == ["CCC.[Na+]"]  # CCO is also in train; mixture components joined
    assert load_dataset("sol", split="test", drop_leaked=False).data == ["CCO", "CCC.[Na+]"]
    everything = load_dataset("sol", targets=["z"])
    assert everything.target.shape == (4, 1) and everything.target_names == ["z"]
    assert everything.info["license"] == "CC0-1.0" and len(set(everything.groups)) == 3


def test_long_format_needs_task_values(hub):
    with pytest.raises(ValueError, match="task_values"):
        load_dataset("assays")
    data = load_dataset("assays", task_values=["A1", "A3"])
    assert data.data == ["CCO", "CCC"] and data.target_names == ["A1", "A3"]
    assert data.target[0, 0] == 6.0 and np.isnan(data.target[0, 1]) and data.target[1, 1] == 5.0


def test_newer_index_schema_asks_for_upgrade(hub):
    path = hub / "molecular-datasets-index" / "index.json"
    index = json.loads(path.read_text())
    path.write_text(json.dumps({**index, "schema": 99}))
    with pytest.raises(RuntimeError, match="Upgrade torch-molecule"):
        list_datasets()


def test_files_only_dataset_names_its_files(hub):
    with pytest.raises(ValueError, match="files/qm7.mat"):
        load_dataset("raw")


def test_link_only_dataset_points_to_source(hub):
    with pytest.raises(ValueError, match="https://example.org/data"):
        load_dataset("closed")


def test_split_keeps_groups_and_info(hub):
    data = load_dataset("sol")
    train, test = data.train_test_split(test_size=0.5, seed=0)
    assert len(train.groups) == len(train.data) and train.info["name"] == "sol"


def test_legacy_loaders_are_deprecated_aliases(monkeypatch):
    from torch_molecule.datasets import load_hf_dataset

    calls = []
    monkeypatch.setattr(load_hf_dataset, "load_dataset", lambda name, **kwargs: calls.append((name, kwargs)) or "data")
    with pytest.warns(FutureWarning, match="load_dataset\\('qm9'\\)"):
        assert load_hf_dataset.load_qm9(local_dir="d") == "data"
    with pytest.warns(FutureWarning):
        assert load_hf_dataset.load_admet(return_local_data_path=True) == ("data", "torchmol_data")
    with pytest.warns(FutureWarning):
        load_hf_dataset.load_gasperm()
    assert [c[0] for c in calls] == ["qm9", "admet_ai", "polymer_gas_permeability"]
    assert calls[0][1] == {"targets": ["gap"], "local_dir": "d"}


def test_list_filters_by_text_split_and_predicate(hub):
    assert [d["name"] for d in list_datasets(search="screening")] == ["screen"]
    rows = list_datasets(split="source")
    assert [d["name"] for d in rows] == ["sol", "screen"] and rows[1]["schemes"] == ["random_1", "scaffold_1"]
    assert [d["name"] for d in list_datasets(where=lambda e: e["tables"] and e["tables"][0]["task_column"])] == ["assays"]


def test_split_schemes_select_rows_by_column(hub):
    default = load_dataset("screen", split="test")
    assert default.data == ["c1ccccc1", "C1CCCC1"] and default.info["scheme"] == "scaffold_1"
    random_test = load_dataset("screen", split="test", scheme="random_1")
    assert random_test.data == ["C1CC1"] and random_test.info["splits"] == ["train", "test"]
    # The unassigned row stays out; C1CC1 is the random_1 test molecule.
    assert load_dataset("screen", split="train", scheme="random_1").data == ["C1CCC1", "c1ccccc1"]
    assert len(load_dataset("screen", scheme="random_1").data) == 3
    with pytest.raises(KeyError, match="schemes"):
        load_dataset("screen", scheme="time_1")
    with pytest.raises(KeyError, match="splits"):
        load_dataset("screen", split="validation", scheme="random_1")


def test_extra_columns_follow_rows_through_splits(hub):
    data = load_dataset("screen", columns=["year"])
    assert data.extra == {"year": [2001, 2005, 2010, 2012]}
    train, test = data.train_test_split(test_size=0.5, seed=0)
    assert sorted(train.extra["year"] + test.extra["year"]) == [2001, 2005, 2010, 2012]
    with pytest.raises(KeyError, match="load_table"):
        load_dataset("screen", columns=["absent"])


def test_raw_tables_observations_and_files(hub):
    rows = load_table("screen", split="test", scheme="random_1")
    assert rows.num_rows == 1 and "year" in rows.column_names
    assert Hub().observations("screen").num_rows == 3
    assert download_files("raw")[0].endswith("files/qm7.mat")
    with pytest.raises(KeyError):
        download_files("raw", patterns=["*.sdf"])
    with pytest.raises(ValueError, match="https://example.org/data"):
        download_files("closed")


def test_hub_reads_an_explicit_index_path(hub, monkeypatch):
    path = str(hub / "molecular-datasets-index" / "index.json")
    monkeypatch.delenv("TORCHMOL_DATASET_INDEX")
    assert Hub(index_path=path).info("sol")["license"] == "CC0-1.0"
