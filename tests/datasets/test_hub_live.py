"""Contract test against the published collection; runs when TORCHMOL_HUB_LIVE=1 (weekly in CI)."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TORCHMOL_HUB_LIVE") != "1", reason="set TORCHMOL_HUB_LIVE=1 to run")


def test_live_index_matches_loader_and_small_datasets_load(tmp_path):
    from torch_molecule.datasets import list_datasets, load_dataset
    from torch_molecule.datasets.hub import _index

    entries = _index(str(tmp_path))
    published = [e for e in entries if e["access"] == "public"]
    assert published, "no public datasets in the index"
    for entry in entries:
        assert {"name", "access", "repo_id", "revision", "links", "quality", "tables"} <= set(entry), entry["name"]
        for table in entry["tables"]:
            assert {"config", "files", "task_column", "targets"} <= set(table), entry["name"]
    assert {row["name"] for row in list_datasets(local_dir=str(tmp_path))} == {e["name"] for e in entries}
    with_table = [e for e in published if e["tables"]]  # some datasets are published as files only
    smallest = sorted((e for e in with_table if not e["tables"][0]["task_column"]), key=lambda e: e["tables"][0]["rows"])[:3]
    for entry in smallest:
        data = load_dataset(entry["name"], local_dir=str(tmp_path))
        assert len(data.data) == len(data.groups) > 0, entry["name"]
