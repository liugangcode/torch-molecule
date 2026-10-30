"""Contract test against the published hub; runs when TORCHMOL_HUB_LIVE=1 (weekly in CI)."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TORCHMOL_HUB_LIVE") != "1", reason="set TORCHMOL_HUB_LIVE=1 to run")


def test_live_index_matches_loader_and_small_datasets_load(tmp_path):
    from torch_molecule.datasets import list_datasets, load_dataset
    from torch_molecule.datasets.hub import _index

    entries = _index(str(tmp_path))["datasets"]
    for entry in entries:
        assert {"name", "access", "repo_id", "revision", "links", "quality", "tables", "default_config"} <= set(entry), entry["name"]
        for table in entry["tables"]:
            assert {"config", "files", "task_column", "targets"} <= set(table), entry["name"]
    assert {row["name"] for row in list_datasets(local_dir=str(tmp_path))} == {e["name"] for e in entries}
    single = [e for e in entries if e["access"] == "public" and len(e["tables"]) == 1 and not e["tables"][0]["task_column"]]
    for entry in sorted(single, key=lambda e: e["tables"][0]["rows"])[:3]:
        data = load_dataset(entry["name"], local_dir=str(tmp_path))
        assert len(data.data) == len(data.groups) > 0, entry["name"]


def test_live_groups_splits_and_former_names(tmp_path):
    from torch_molecule.datasets import load_dataset

    train = load_dataset("moleculeace", "chembl204_ki", split="train", local_dir=str(tmp_path))
    test = load_dataset("moleculeace", "chembl204_ki", split="test", local_dir=str(tmp_path))
    assert not set(train.groups) & set(test.groups)
    assert train.info["targets"]["y [pEC50/pKi]"]["label"] == "Prothrombin · pKi"
    assert load_dataset("moleculeace_chembl204_ki", split="train", local_dir=str(tmp_path)).data == train.data
    with pytest.raises(ValueError, match="configs"):
        load_dataset("moleculeace", local_dir=str(tmp_path))
    scaffold = load_dataset("welqrate", "aid1798", split="test", scheme="scaffold_2", local_dir=str(tmp_path))
    assert scaffold.info["scheme"] == "scaffold_2" and len(scaffold.data) > 0
