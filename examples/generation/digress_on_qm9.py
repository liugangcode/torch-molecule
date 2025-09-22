import os

from torch_molecule import DigressMolecularGenerator
from torch_molecule.datasets import load_qm9


def train_on_qm9() -> None:
    model = DigressMolecularGenerator(verbose="progress_bar", batch_size=1024, epochs=2)

    smiles_list, _ = load_qm9(local_dir="torchmol_data")

    original_count = len(smiles_list)
    smiles_list = [s for s in smiles_list if isinstance(s, str) and s]
    if original_count > len(smiles_list):
        print(f"Data cleaning: removed {original_count - len(smiles_list)} invalid entries from QM9 dataset.")

    model.fit(smiles_list)

    print("\n=== Generating 10 molecules from QM9-trained model ===")
    generated_smiles = model.generate(batch_size=10)
    print(f"Generated {len(generated_smiles)} molecules.")
    for i, smiles in enumerate(generated_smiles, start=1):
        print(f"{i}: {smiles}")


if __name__ == "__main__":
    train_on_qm9()
