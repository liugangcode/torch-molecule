import os
import shutil

from torch_molecule import HFPretrainedMolecularGenerator

REPO_ID = "chandar-lab/NovoMolGen_32M_SMILES_BPE"
N_SAMPLES = 5
TRAIN_SMILES = [
    "CC(=O)O",
    "CCO",
    "CCCC",
    "c1ccccc1",
    "CCN",
]


def test_novomolgen_generator():
    print("\n=== Testing NovoMolGen initialization ===")
    model = HFPretrainedMolecularGenerator(
        repo_id=REPO_ID,
        verbose="progress_bar",
    )
    print("NovoMolGen initialized successfully")

    print("\n=== Testing NovoMolGen loading from Hugging Face ===")
    model.fit()
    print("NovoMolGen loaded successfully")

    print("\n=== Testing NovoMolGen de novo generation ===")
    generated_smiles = model.generate(n_samples=N_SAMPLES)
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing NovoMolGen saving and loading ===")
    save_path = "pretrained_novomolgen_test_model"
    model.save_to_local(save_path)
    print(f"NovoMolGen saved to {save_path}")

    loaded_model = HFPretrainedMolecularGenerator(repo_id=REPO_ID)
    loaded_model.load_from_local(save_path)
    print("NovoMolGen loaded from local directory")

    generated_smiles = loaded_model.generate(n_samples=2)
    print(f"Generated {len(generated_smiles)} molecules with loaded model")
    print("Example generated SMILES:", generated_smiles[:2])

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Cleaned up {save_path}")

    print("\n=== Testing NovoMolGen fine-tuning ===")
    finetune_model = HFPretrainedMolecularGenerator(
        repo_id=REPO_ID,
        batch_size=2,
        epochs=1,
        verbose="progress_bar",
    )
    finetune_model.fit(TRAIN_SMILES)
    print("Fine-tuning completed")
    print(f"Fitting epochs: {finetune_model.fitting_epoch + 1}")
    print(f"Fitting loss: {finetune_model.fitting_loss}")

    generated_smiles = finetune_model.generate(n_samples=2)
    print(f"Generated {len(generated_smiles)} molecules after fine-tuning")
    print("Example generated SMILES:", generated_smiles[:2])

    print("\n=== Testing fine-tuned NovoMolGen saving and loading ===")
    save_path = "pretrained_novomolgen_finetune_test_model"
    finetune_model.save_to_local(save_path)
    print(f"Fine-tuned NovoMolGen saved to {save_path}")

    loaded_finetune = HFPretrainedMolecularGenerator(repo_id=REPO_ID)
    loaded_finetune.load_from_local(save_path)
    print("Fine-tuned NovoMolGen loaded from local directory")

    generated_smiles = loaded_finetune.generate(n_samples=2)
    print(f"Generated {len(generated_smiles)} molecules with loaded fine-tuned model")
    print("Example generated SMILES:", generated_smiles[:2])

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Cleaned up {save_path}")


if __name__ == "__main__":
    test_novomolgen_generator()
