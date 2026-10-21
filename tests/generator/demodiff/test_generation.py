import pytest
import torch
from rdkit import Chem


@pytest.mark.parametrize("conditioned", [False, True])
def test_sampling_runs_diffusion_and_returns_valid_molecules(make_model, smiles, conditioned):
    torch.manual_seed(29)
    model = make_model()
    model.fit(smiles)
    forwards = []

    def inspect_score(_module, args, _output):
        score, masks = args[3:5]
        target_mask = masks["target"]
        target_width = target_mask.shape[1]
        assert torch.all(score[:, :target_width, 0][target_mask] == 1)
        forwards.append(
            (score.detach()[masks["all"]].clone(), masks["all"].sum(), target_mask.sum())
        )

    hook = model.model.model.register_forward_hook(inspect_score)
    kwargs = {"contexts": ["CCO"], "score": [0.2]} if conditioned else {}
    try:
        molecules = model.generate(3, num_nodes=1, **kwargs)
    finally:
        hook.remove()
    assert forwards
    assert any(all_nodes == target_nodes for _, all_nodes, target_nodes in forwards)
    if conditioned:
        assert any(all_nodes > target_nodes for _, all_nodes, target_nodes in forwards)
        assert any(torch.any(torch.isclose(score, torch.tensor(0.2))) for score, _, _ in forwards)
    assert len(molecules) == 3
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in molecules)


def test_bounded_sampling_warns_and_returns_available_results(make_model, smiles, monkeypatch):
    model = make_model()
    model.fit(smiles)
    monkeypatch.setattr(model.tokenizer, "decode", lambda *_: None)
    with pytest.warns(UserWarning, match="0.*3|max_attempts|fewer"):
        assert model.generate(3, num_nodes=1, max_attempts=1) == []


def test_generation_respects_the_context_length_budget(make_model, smiles):
    model = make_model(context_length=2)
    model.fit(smiles)
    with pytest.raises(ValueError, match="context_length"):
        model.generate(1, contexts=["CCO", "CCN"], score=[0.2, 0.4], num_nodes=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contexts": ["CCO"]},
        {"score": [0.2]},
        {"contexts": ["CCO"], "score": []},
        *[
            {"contexts": ["CCO"], "score": [value]}
            for value in [-0.01, 1.01, float("nan"), float("inf")]
        ],
    ],
)
def test_generate_rejects_missing_mismatched_or_invalid_similarity_scores(
    make_model, smiles, kwargs
):
    model = make_model()
    model.fit(smiles[:2])
    with pytest.raises(ValueError, match="score"):
        model.generate(1, **kwargs)


def test_generate_accepts_similarity_endpoints(make_model, smiles):
    torch.manual_seed(37)
    model = make_model()
    model.fit(smiles)
    molecules = model.generate(1, contexts=["CCO", "CCN"], score=[0.0, 1.0], num_nodes=1)
    assert len(molecules) == 1
    assert Chem.MolFromSmiles(molecules[0]) is not None
