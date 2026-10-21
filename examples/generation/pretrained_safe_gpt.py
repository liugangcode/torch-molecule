import os
import shutil

from torch_molecule import HFPretrainedMolecularGenerator

REPO_ID = "datamol-io/safe-gpt"
N_SAMPLES = 5
SHORT_SCAFFOLD = "c1ccccc1"
LONG_SCAFFOLD = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"


def test_safe_gpt_generator():
    print("\n=== Testing SAFE-GPT initialization ===")
    model = HFPretrainedMolecularGenerator(
        repo_id=REPO_ID,
        verbose="progress_bar",
    )
    print("SAFE-GPT initialized successfully")

    print("\n=== Testing SAFE-GPT loading from Hugging Face ===")
    model.fit()
    print("SAFE-GPT loaded successfully")

    print("\n=== Testing SAFE-GPT de novo generation ===")
    generated_smiles = model.generate(n_samples=N_SAMPLES)
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing SAFE-GPT short scaffold generation ===")
    generated_smiles = model.generate(n_samples=N_SAMPLES, scaffold=SHORT_SCAFFOLD)
    print(f"Generated {len(generated_smiles)} molecules from scaffold {SHORT_SCAFFOLD}")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing SAFE-GPT long scaffold generation ===")
    generated_smiles = model.generate(n_samples=2, scaffold=LONG_SCAFFOLD)
    print(f"Generated {len(generated_smiles)} molecules from long scaffold")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing SAFE-GPT saving and loading ===")
    save_path = "pretrained_safe_gpt_test_model"
    model.save_to_local(save_path)
    print(f"SAFE-GPT saved to {save_path}")

    loaded_model = HFPretrainedMolecularGenerator(repo_id=REPO_ID)
    loaded_model.load_from_local(save_path)
    print("SAFE-GPT loaded from local directory")

    generated_smiles = loaded_model.generate(n_samples=2, scaffold=SHORT_SCAFFOLD)
    print(f"Generated {len(generated_smiles)} molecules with loaded model")
    print("Example generated SMILES:", generated_smiles[:2])

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Cleaned up {save_path}")


if __name__ == "__main__":
    test_safe_gpt_generator()
