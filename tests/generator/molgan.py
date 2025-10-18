import os
from rdkit import RDLogger
from torch_molecule.generator.molgan import (
    MolGAN,
    RewardOracle,
)

RDLogger.DisableLog("rdApp.*")

def test_molgan():
    # Sample SMILES list
    smiles_list = [
        "CCO", "CCN", "CCC", "COC",
        "CCCl", "CCF", "CBr", "CN(C)C", "CC(=O)O", "c1ccccc1",
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    model_decoder = ["C", "N", "O", "F", "Cl", "Br"]

    # 1. Initialize MolGAN
    print("\n=== Testing MolGAN Initialization ===")
    GANConfig = {
        "num_nodes": 9,
        "num_layers": 4,
        "num_atom_types": 5,
        "num_bond_types": 4,
        "latent_dim": 56,
        "hidden_dims_gen": [128, 128],
        "hidden_dims_disc": [128, 128],
        "tau": 1.0,
        "use_reward": True,
    }
    model = MolGAN(**GANConfig, device="cpu")
    print("MolGAN initialized successfully")

    # 2. Fit with QED reward
    print("\n=== Testing MolGAN Training with QED Reward ===")
    reward = RewardOracle(kind="qed")
    model.fit(X=smiles_list, reward=reward, epochs=5, batch_size=16)
    print("MolGAN trained successfully")

    # 3. Generation
    print("\n=== Testing MolGAN Generation ===")
    gen_smiles = model.generate(n_samples=10)
    print(f"Generated {len(gen_smiles)} SMILES")
    print("Example generated molecules:", gen_smiles[:3])

    # 4. Save and Reload
    print("\n=== Testing MolGAN Save & Load ===")
    save_dir = "molgan-test"
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    model2 = MolGAN.from_pretrained(save_dir)
    print("Model loaded successfully")

    gen_smiles2 = model2.generate(n_samples=5)
    print("Generated after loading:", gen_smiles2[:3])

    # 5. Cleanup
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Cleaned up {save_dir}")

if __name__ == "__main__":
    test_molgan()
