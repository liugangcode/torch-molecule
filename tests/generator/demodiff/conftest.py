import pytest
import torch

from torch_molecule import DemoDiffMolecularGenerator


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def smiles():
    return ["CCO", "CCN", "CCC", "CC(=O)Oc1ccccc1C(=O)O"]


@pytest.fixture
def make_model():
    def create(**kwargs):
        params = dict(
            num_layer=1,
            hidden_size=32,
            num_head=4,
            diffusion_steps=2,
            context_length=64,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            device="cpu",
        )
        params.update(kwargs)
        return DemoDiffMolecularGenerator(**params)

    return create
