import runpy
from pathlib import Path

import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def test_example_uses_similarity_scores_and_selects_the_most_similar_demonstrations():
    example = Path(__file__).resolve().parents[3] / "examples" / "generation" / "demodiff.py"
    build_demonstrations = runpy.run_path(str(example))["build_demonstrations"]
    smiles = ["CCO", "CCO", "CCN", "c1ccccc1"]
    contexts, score = build_demonstrations(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]
    assert contexts[0][0] == 1
    assert score[0][0] == 1.0
    for target, (indices, values) in enumerate(zip(contexts, score)):
        expected = {
            index: DataStructs.TanimotoSimilarity(fingerprints[target], fingerprint)
            for index, fingerprint in enumerate(fingerprints)
            if index != target
        }
        assert target not in indices
        assert values == pytest.approx([expected[index] for index in indices])
        assert values == sorted(expected.values(), reverse=True)[:2]
