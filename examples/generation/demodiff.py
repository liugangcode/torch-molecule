"""Train a small DemoDiff model on local SMILES, with optional demonstrations.

Run from the project root in the torch-molecule mamba environment:
    python examples/generation/demodiff.py
    python examples/generation/demodiff.py --with-context
    python examples/generation/demodiff.py --data data/polymer100.csv --with-context

Demonstration score uses Morgan fingerprint Tanimoto similarity in [0, 1]:
1 for identical fingerprints, 0 for no shared fingerprint bits. This is a local
example of normalized task scores; the target's score is always 1 internally.
Supply the same score definition to fit() and generate(). Scores describe the
demonstrations, and do not guarantee measured similarities of generated molecules.
Only the smiles column is used. Generation does not enforce polymer attachment
atoms, and the tokenizer can lose stereochemistry and formal charges.
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from torch_molecule import DemoDiffMolecularGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_demonstrations(smiles):
    """Choose two most similar molecules; score is Tanimoto similarity in [0, 1]."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]
    contexts, score = [], []
    for target, fingerprint in enumerate(fingerprints):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        indices = sorted(
            (index for index in range(len(smiles)) if index != target),
            key=lambda index: similarity[index],
            reverse=True,
        )[:2]
        contexts.append(indices)
        score.append([similarity[index] for index in indices])
    return contexts, score


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "molecule100.csv")
    parser.add_argument(
        "--with-context", action="store_true", help="Train with molecular demonstrations."
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--device", default=None, help="For example: cpu, cuda, or cuda:1.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    with args.data.open(newline="") as handle:
        smiles = [row["smiles"] for row in csv.DictReader(handle)]

    mode = "conditioned" if args.with_context else "unconditional"
    output = (
        PROJECT_ROOT / "artifacts" / "demodiff" / "examples_similarity" / f"{args.data.stem}_{mode}"
    )
    output.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(smiles)} molecules from {args.data} ({mode}).")

    model = DemoDiffMolecularGenerator(
        num_layer=2,
        hidden_size=64,
        num_head=4,
        mlp_ratio=2,
        diffusion_steps=20,
        batch_size=16,
        epochs=args.epochs,
        learning_rate=1e-3,
        drop_context=0.1 if args.with_context else 0.0,
        guidance_scale=2.0 if args.with_context else 1.0,
        device=args.device,
        verbose="progress_bar",
    )
    if args.with_context:
        # fit() uses training indices, with one similarity score per demonstration.
        contexts, score = build_demonstrations(smiles)
        print(f"First target: contexts={contexts[0]}, score={score[0]}")
        model.fit(smiles, contexts=contexts, score=score)
    else:
        model.fit(smiles)

    checkpoint = str(output / "model.pt")
    model.save_to_local(checkpoint)
    loaded = DemoDiffMolecularGenerator(device=model.device)
    loaded.load_from_local(checkpoint)

    if args.with_context:
        # Reuse the first target's demonstrations and their similarity scores.
        generated = loaded.generate(
            n_samples=args.n_samples,
            contexts=[smiles[index] for index in contexts[0]],
            score=score[0],
        )
    else:
        generated = loaded.generate(n_samples=args.n_samples)

    with (output / "generated.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        writer.writerows([value] for value in generated)
    with (output / "loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss"])
        writer.writerows(enumerate(model.fitting_loss, start=1))

    print(f"Returned {len(generated)}/{args.n_samples} valid connected molecules.")
    print(f"Outputs: {output}")
    for value in generated[:5]:
        print(value)


if __name__ == "__main__":
    main()
