import torch

from torch_molecule import DemoDiffMolecularGenerator


def test_checkpoint_restores_configuration_weights_and_generation(make_model, smiles, tmp_path):
    torch.manual_seed(23)
    original = make_model()
    original.fit(smiles, contexts=[[1], [2], [3], [0]], score=[[0.0], [0.3], [0.7], [1.0]])
    path = str(tmp_path / "demodiff.pt")
    original.save_to_local(path)
    restored = DemoDiffMolecularGenerator(device="cpu")
    restored.load_from_local(path)
    assert restored.is_fitted_
    assert restored.hidden_size == original.hidden_size
    assert restored.dataset_info_ == original.dataset_info_
    assert restored.fitting_loss == original.fitting_loss
    assert any("score_embedder" in name for name in original.model.state_dict())
    for name, expected in original.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, rtol=0, atol=0)
    torch.manual_seed(31)
    expected = original.generate(3, num_nodes=1, contexts=["CCO"], score=[0.8])
    torch.manual_seed(31)
    actual = restored.generate(3, num_nodes=1, contexts=["CCO"], score=[0.8])
    assert actual == expected
    assert len(actual) == 3
