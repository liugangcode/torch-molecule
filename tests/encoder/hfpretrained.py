import numpy as np
import torch
from torch_molecule import HFPretrainedMolecularEncoder

def test_hf_pretrained_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    # Test different HuggingFace models
    models_to_test = [
        {"repo_id": "entropy/gpt2_zinc_87m", "model_name": "GPT-2_ZINC_87M"},
        {"repo_id": "entropy/roberta_zinc_480m", "model_name": "RoBERTa_ZINC_480M"},
        {"repo_id": "ncfrey/ChemGPT-1.2B", "model_name": "ChemGPT_1.2B"},
        {"repo_id": "ncfrey/ChemGPT-19M", "model_name": "ChemGPT_19M"},
        {"repo_id": "ncfrey/ChemGPT-4.7M", "model_name": "ChemGPT_4.7M"},
        {"repo_id": "DeepChem/ChemBERTa-77M-MTR", "model_name": "ChemBERTa_77M_MTR"},
        {"repo_id": "DeepChem/ChemBERTa-77M-MLM", "model_name": "ChemBERTa_77M_MLM"},
        {"repo_id": "DeepChem/ChemBERTa-10M-MTR", "model_name": "ChemBERTa_10M_MTR"},
        {"repo_id": "DeepChem/ChemBERTa-10M-MLM", "model_name": "ChemBERTa_10M_MLM"},
        {"repo_id": "DeepChem/ChemBERTa-5M-MLM", "model_name": "ChemBERTa_5M_MLM"},
        {"repo_id": "DeepChem/ChemBERTa-5M-MTR", "model_name": "ChemBERTa_5M_MTR"}
        {"repo_id": "seyonec/ChemBERTa-zinc-base-v1", "model_name": "ChemBERTa_zinc_base_v1"},
        {"repo_id": "unikei/bert-base-smiles", "model_name": "bert-base-smiles"}
    ]

    for model_config in models_to_test:
        print(f"\n=== Testing {model_config['model_name']} ===")
        
        # Initialize model
        model = HFPretrainedMolecularEncoder(repo_id=model_config["repo_id"], model_name=model_config["model_name"])
        print(f"Model initialized successfully: {model_config['model_name']}")
        
        # Load the model
        print("Loading model from HuggingFace...")
        model.fit()
        print("Model loaded successfully")
        
        # Encoding test
        print("Testing molecule encoding...")
        encodings_pt = model.encode(molecules, return_type="pt")
        encodings_np = model.encode(molecules, return_type="np")
        
        print('model_config', model_config)
        print(f"Encoded {len(molecules)} molecules")
        print(f"PyTorch tensor shape: {encodings_pt.shape}")
        print(f"NumPy array shape: {encodings_np.shape}")
        
        # Verify PyTorch and NumPy outputs match
        if np.allclose(encodings_pt.cpu().numpy(), encodings_np):
            print("PyTorch and NumPy encodings match!")
        else:
            print("Warning: PyTorch and NumPy encodings differ")
        
        # Print some stats about the embeddings
        print(f"Embedding dimensionality: {encodings_pt.shape[1]}")
        print(f"Mean embedding value: {encodings_pt.mean().item():.4f}")
        print(f"Std of embedding values: {encodings_pt.std().item():.4f}")
        
        # Check if embeddings are different for different molecules
        distances = []
        for i in range(len(molecules)):
            for j in range(i+1, len(molecules)):
                dist = torch.norm(encodings_pt[i] - encodings_pt[j]).item()
                distances.append(dist)
        
        print(f"Average L2 distance between embeddings: {np.mean(distances):.4f}")
        print(f"Min L2 distance between embeddings: {np.min(distances):.4f}")
        print(f"Max L2 distance between embeddings: {np.max(distances):.4f}")

if __name__ == "__main__":
    test_hf_pretrained_encoder()