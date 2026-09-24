import os
import shutil

from torch_molecule import HFPretrainedMolecularGenerator

REPO_ID = "fairydance/molexar-10m-base"
N_SAMPLES = 2
START_SMILES = "[*]C1(CC#N)CN(S(=O)(=O)CC)C1"


def test_molexar_generator():
    print("\n=== Testing Molexar initialization ===")
    model = HFPretrainedMolecularGenerator(
        repo_id=REPO_ID,
        verbose="progress_bar",
    )
    print("Molexar initialized successfully")

    print("\n=== Testing Molexar loading from Hugging Face ===")
    model.fit()
    print("Molexar loaded successfully")

    print("\n=== Testing Molexar de novo generation ===")
    generated_smiles = model.generate(
        n_samples=N_SAMPLES,
        max_new_tokens=64,
        temperature=0.8,
    )
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing Molexar fragment-constrained generation ===")
    generated_smiles = model.generate(
        n_samples=N_SAMPLES,
        start_smiles=START_SMILES,
        generation_task="motif_extension",
        max_new_tokens=64,
        temperature=0.8,
    )
    print(f"Generated {len(generated_smiles)} molecules from start_smiles {START_SMILES}")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing Molexar saving and loading ===")
    save_path = "pretrained_molexar_test_model"
    model.save_to_local(save_path)
    print(f"Molexar saved to {save_path}")

    loaded_model = HFPretrainedMolecularGenerator(repo_id=REPO_ID)
    loaded_model.load_from_local(save_path)
    print("Molexar loaded from local directory")

    generated_smiles = loaded_model.generate(
        n_samples=N_SAMPLES,
        max_new_tokens=64,
        temperature=0.8,
    )
    print(f"Generated {len(generated_smiles)} molecules with loaded model")
    print("Example generated SMILES:", generated_smiles[:2])

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Cleaned up {save_path}")


if __name__ == "__main__":
    test_molexar_generator()
