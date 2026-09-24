import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_molecule.datasets import SMILESDataset, subsample, train_test_split
from torch_molecule.datasets.split import _BUTINA_OOM_MESSAGE


def _benzene_family():
    return [
        "c1ccccc1",
        "c1ccccc1O",
        "c1ccccc1N",
        "c1ccccc1C",
        "c1ccccc1Cl",
        "c1ccccc1F",
        "Cc1ccccc1O",
        "Nc1ccccc1O",
    ]


def _other_molecules():
    return [
        "CCO",
        "CCN",
        "CCC",
        "C1CCCCC1",
        "n1ccccc1",
        "C1CCNCC1",
        "CC(=O)O",
        "CC(C)O",
    ]


def _labeled_dataset():
    smiles = _benzene_family() + _other_molecules()
    y = np.arange(len(smiles), dtype=np.float32).reshape(-1, 1)
    return SMILESDataset(data=smiles, target=y)


def _scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(scaffold)


def test_random_split_reproducible():
    data = _labeled_dataset()
    train_a, test_a = train_test_split(data, test_size=0.25, method="random", seed=42)
    train_b, test_b = train_test_split(data, test_size=0.25, method="random", seed=42)
    assert train_a.data == train_b.data
    assert test_a.data == test_b.data
    np.testing.assert_array_equal(train_a.target, train_b.target)


def test_random_split_different_seeds():
    data = _labeled_dataset()
    train_a, _ = train_test_split(data, test_size=0.25, method="random", seed=0)
    train_b, _ = train_test_split(data, test_size=0.25, method="random", seed=1)
    assert train_a.data != train_b.data


def test_random_split_ratio_and_coverage():
    data = _labeled_dataset()
    train, holdout = train_test_split(data, test_size=0.25, method="random", seed=42)
    n = len(data.data)
    assert len(train.data) + len(holdout.data) == n
    assert abs(len(holdout.data) / n - 0.25) < 1e-9
    assert set(train.data).isdisjoint(holdout.data)
    assert set(train.data) | set(holdout.data) == set(data.data)


def test_random_preserves_multitask_target():
    smiles = _benzene_family() + _other_molecules()
    y = np.column_stack(
        [
            np.arange(len(smiles), dtype=np.float32),
            np.arange(len(smiles), dtype=np.float32) * 2,
        ]
    )
    data = SMILESDataset(data=smiles, target=y)
    train, holdout = train_test_split(data, test_size=0.2, method="random", seed=7)
    assert train.target.shape[1] == 2
    assert holdout.target.shape[1] == 2
    assert train.target.shape[0] == len(train.data)


def test_split_unlabeled_dataset():
    data = SMILESDataset(data=_benzene_family() + _other_molecules(), target=None)
    train, holdout = train_test_split(data, test_size=0.2, method="random", seed=1)
    assert train.target is None
    assert holdout.target is None
    assert len(train.data) + len(holdout.data) == len(data.data)


def test_scaffold_no_leakage():
    data = _labeled_dataset()
    train, holdout = train_test_split(data, test_size=0.3, method="scaffold", seed=42)
    train_scaffolds = {_scaffold_smiles(s) for s in train.data}
    holdout_scaffolds = {_scaffold_smiles(s) for s in holdout.data}
    assert train_scaffolds.isdisjoint(holdout_scaffolds)
    assert len(train.data) + len(holdout.data) == len(data.data)
    assert len(set(train.data) & set(holdout.data)) == 0


def test_scaffold_keeps_benzene_family_together():
    data = _labeled_dataset()
    train, holdout = train_test_split(data, test_size=0.3, method="scaffold")
    benzene_scaffold = _scaffold_smiles("c1ccccc1")
    train_has = any(_scaffold_smiles(s) == benzene_scaffold for s in train.data)
    holdout_has = any(_scaffold_smiles(s) == benzene_scaffold for s in holdout.data)
    assert train_has ^ holdout_has


def test_scaffold_acyclic_molecules_do_not_crash():
    data = SMILESDataset(
        data=["CCO", "CCN", "CCC", "CC", "C"],
        target=np.arange(5).reshape(-1, 1),
    )
    train, holdout = train_test_split(data, test_size=0.4, method="scaffold")
    assert len(train.data) >= 1
    assert len(holdout.data) >= 1


def test_scaffold_invalid_smiles_raises():
    data = SMILESDataset(data=["CCO", "not_a_smiles"], target=None)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        train_test_split(data, method="scaffold")


def test_unknown_method_raises():
    data = _labeled_dataset()
    with pytest.raises(ValueError, match="Unknown split method"):
        train_test_split(data, method="kmeans")


def test_subsample_reproducible_and_size():
    data = _labeled_dataset()
    a = subsample(data, n=5, seed=0)
    b = subsample(data, n=5, seed=0)
    c = subsample(data, n=5, seed=1)
    assert len(a.data) == 5
    assert a.data == b.data
    assert a.data != c.data
    assert a.target.shape == (5, 1)


def test_subsample_and_split_methods_on_dataset():
    data = _labeled_dataset()
    small = data.subsample(n=10, seed=3)
    assert len(small.data) == 10
    train, holdout = small.train_test_split(test_size=0.3, method="random", seed=4)
    assert len(train.data) + len(holdout.data) == 10
    butina_train, butina_hold = small.train_test_split(
        test_size=0.3, method="butina", similarity_cutoff=0.4
    )
    assert len(butina_train.data) + len(butina_hold.data) == 10
    size_train, size_hold = small.train_test_split(test_size=0.3, method="size")
    assert len(size_train.data) + len(size_hold.data) == 10


def test_subsample_rejects_too_large_n():
    data = _labeled_dataset()
    with pytest.raises(ValueError, match="larger than the dataset size"):
        data.subsample(n=len(data.data) + 1)


def test_target_row_mismatch_raises():
    data = SMILESDataset(data=["CCO", "CCC"], target=np.array([[1.0]]))
    with pytest.raises(ValueError, match="target has"):
        train_test_split(data, method="random")


def _rdkit_butina_clusters(smiles, similarity_cutoff=0.65):
    from rdkit.ML.Cluster import Butina
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs as RDS

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    dists = []
    for i in range(1, len(fps)):
        sims = RDS.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    return Butina.ClusterData(
        dists, len(fps), 1.0 - similarity_cutoff, isDistData=True
    )


def _cluster_membership(clusters):
    return {frozenset(cluster) for cluster in clusters}


def test_butina_matches_rdkit_clusterdata_membership():
    from torch_molecule.datasets.split import _butina_clusters

    smiles = _benzene_family() + _other_molecules()
    for cutoff in (0.3, 0.4, 0.65):
        ours = _butina_clusters(smiles, similarity_cutoff=cutoff)
        rdkit = _rdkit_butina_clusters(smiles, similarity_cutoff=cutoff)
        assert _cluster_membership(ours) == _cluster_membership(rdkit)
        for cluster, ref in zip(ours, rdkit):
            assert cluster[0] == ref[0]


def test_butina_no_cluster_leakage():
    from torch_molecule.datasets.split import _butina_clusters

    data = _labeled_dataset()
    cutoff = 0.4
    train, holdout = train_test_split(
        data, test_size=0.3, method="butina", similarity_cutoff=cutoff
    )
    smiles_to_idx = {s: i for i, s in enumerate(data.data)}
    clusters = _butina_clusters(data.data, similarity_cutoff=cutoff)
    train_idx = {smiles_to_idx[s] for s in train.data}
    holdout_idx = {smiles_to_idx[s] for s in holdout.data}
    assert train_idx.isdisjoint(holdout_idx)
    assert train_idx | holdout_idx == set(range(len(data.data)))
    for cluster in clusters:
        members = set(cluster)
        assert members <= train_idx or members <= holdout_idx


def test_butina_ignores_seed_and_is_reproducible():
    data = _labeled_dataset()
    a_train, a_hold = train_test_split(
        data, test_size=0.3, method="butina", seed=0, similarity_cutoff=0.4
    )
    b_train, b_hold = train_test_split(
        data, test_size=0.3, method="butina", seed=1, similarity_cutoff=0.4
    )
    assert a_train.data == b_train.data
    assert a_hold.data == b_hold.data


def test_butina_invalid_smiles_raises():
    data = SMILESDataset(data=["CCO", "not_a_smiles"], target=None)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        train_test_split(data, method="butina")


def test_butina_rejects_bad_cutoff():
    data = _labeled_dataset()
    with pytest.raises(ValueError, match="similarity_cutoff"):
        train_test_split(data, method="butina", similarity_cutoff=0.0)
    with pytest.raises(ValueError, match="similarity_cutoff"):
        train_test_split(data, method="butina", similarity_cutoff=1.5)


def _heavy_atoms(smiles: str) -> int:
    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


def test_size_split_holds_out_larger_molecules():
    smiles = ["C", "CC", "CCC", "CCCC", "c1ccccc1", "c1ccccc1c1ccccc1"]
    y = np.arange(len(smiles), dtype=np.float32).reshape(-1, 1)
    data = SMILESDataset(data=smiles, target=y)
    train, holdout = train_test_split(
        data, test_size=1 / 3, method="size", direction="small_to_large"
    )
    assert len(holdout.data) == 2
    assert max(_heavy_atoms(s) for s in train.data) <= min(
        _heavy_atoms(s) for s in holdout.data
    )
    assert set(train.data) | set(holdout.data) == set(smiles)


def test_size_split_large_to_small_holds_out_smaller_molecules():
    smiles = ["C", "CC", "CCC", "CCCC", "c1ccccc1", "c1ccccc1c1ccccc1"]
    data = SMILESDataset(data=smiles, target=None)
    train, holdout = train_test_split(
        data, test_size=1 / 3, method="size", direction="large_to_small"
    )
    assert max(_heavy_atoms(s) for s in holdout.data) <= min(
        _heavy_atoms(s) for s in train.data
    )


def test_size_split_sizeshiftreg_protocol():
    smiles = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CCCCCCC",
        "CCCCCCCC",
        "CCCCCCCCC",
        "c1ccccc1",
    ]
    data = SMILESDataset(data=smiles, target=None)
    train, holdout = train_test_split(
        data, test_size=0.2, method="size", mode="sizeshiftreg"
    )
    n = len(smiles)
    assert len(train.data) == int(round(0.5 * n))
    assert len(holdout.data) == int(round(0.1 * n))
    assert len(train.data) + len(holdout.data) < n
    assert max(_heavy_atoms(s) for s in train.data) <= min(
        _heavy_atoms(s) for s in holdout.data
    )


def test_size_invalid_smiles_raises():
    data = SMILESDataset(data=["CCO", "not_a_smiles"], target=None)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        train_test_split(data, method="size")


def test_butina_oom_does_not_suggest_subsample(monkeypatch):
    data = _labeled_dataset()

    def _boom(*args, **kwargs):
        raise MemoryError("Unable to allocate array")

    monkeypatch.setattr(
        "torch_molecule.datasets.split._butina_groups", _boom
    )
    with pytest.raises(MemoryError, match="Do not subsample") as excinfo:
        train_test_split(data, method="butina")
    assert "chemfp" in str(excinfo.value)
    assert "Do not subsample" in _BUTINA_OOM_MESSAGE
