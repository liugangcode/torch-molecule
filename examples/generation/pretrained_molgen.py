import os
import shutil

from torch_molecule import HFPretrainedMolecularGenerator

N_SAMPLES = 2
SCAFFOLD = "c1ccccc1"
PREFIX_SELFIES = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"


def test_molgen_generator():
    models_to_test = [
        {"repo_id": "zjunlp/MolGen-large", "model_name": "MolGen-large"},
        {"repo_id": "zjunlp/MolGen-large-opt", "model_name": "MolGen-large-opt"},
    ]

    for model_config in models_to_test:
        name = model_config["model_name"]
        repo_id = model_config["repo_id"]

        print(f"\n=== Testing {name} initialization ===")
        model = HFPretrainedMolecularGenerator(
            repo_id=repo_id,
            verbose="progress_bar",
        )
        print(f"{name} initialized successfully")

        print(f"\n=== Testing {name} loading from Hugging Face ===")
        model.fit()
        print(f"{name} loaded successfully")

        print(f"\n=== Testing {name} de novo generation ===")
        generated_smiles = model.generate(n_samples=N_SAMPLES, num_beams=5)
        print(f"Generated {len(generated_smiles)} molecules")
        print("Example generated SMILES:", generated_smiles[:2])

        print(f"\n=== Testing {name} scaffold generation ===")
        generated_smiles = model.generate(
            n_samples=N_SAMPLES,
            scaffold=SCAFFOLD,
            num_beams=5,
        )
        print(f"Generated {len(generated_smiles)} molecules from scaffold {SCAFFOLD}")
        print("Example generated SMILES:", generated_smiles[:2])

        print(f"\n=== Testing {name} prefix_selfies generation ===")
        generated_smiles = model.generate(
            n_samples=N_SAMPLES,
            prefix_selfies=PREFIX_SELFIES,
            num_beams=5,
        )
        print(f"Generated {len(generated_smiles)} molecules from prefix_selfies")
        print("Example generated SMILES:", generated_smiles[:2])

        print(f"\n=== Testing {name} saving and loading ===")
        save_path = f"pretrained_{name.lower().replace('-', '_')}_test_model"
        model.save_to_local(save_path)
        print(f"{name} saved to {save_path}")

        loaded_model = HFPretrainedMolecularGenerator(repo_id=repo_id)
        loaded_model.load_from_local(save_path)
        print(f"{name} loaded from local directory")

        generated_smiles = loaded_model.generate(n_samples=N_SAMPLES, num_beams=5)
        print(f"Generated {len(generated_smiles)} molecules with loaded model")
        print("Example generated SMILES:", generated_smiles[:2])

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print(f"Cleaned up {save_path}")


if __name__ == "__main__":
    test_molgen_generator()
