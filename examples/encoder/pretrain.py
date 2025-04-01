import pandas as pd
import numpy as np
from torch_molecule import AttrMaskMolecularEncoder, ContextPredMolecularEncoder, EdgePredMolecularEncoder, MoamaMolecularEncoder
import os
import argparse

def download_dataset_from_hub(repo_id: str, filename: str, path: str):
    """Load dataset file from Hugging Face Hub, saving locally to `path`."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub package is required to load from Hugging Face Hub. "
            "Install it with: pip install huggingface_hub"
        )

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            #local_dir=os.path.dirname(path),
            repo_type="dataset"  # Important: specify that this is a dataset repo
        )

        return downloaded_path

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset file: {e}")

def train_model(model_type='attrmask'):
    """
    Train encoder models on the training dataset.
    
    Args:
        model_type (str): Type of model to train - 'attrmask', 'contextpred', 'edgepred', 'moama'
    """
    
    # Download dataset from Hugging Face Hub
    local_file = download_dataset_from_hub(
        repo_id="Einae/Zinc_standard_agent",
        filename="smiles.csv",
        path="./data/smiles.csv"
        )

    print('Successfully downloaded the dataset from Hugging Face Hub.')
    
    # Load data
    train_data = pd.read_csv(local_file)
    
    # Extract SMILES and properties
    train_smiles_list = train_data['smiles'].tolist()
    
    # Uncomment the following line if you want to verify training on a small subset
    train_smiles_list = train_smiles_list[:100]
    
    # Create output directory
    output_dir = f"output/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the specified model
    if model_type == 'attrmask':
        model = AttrMaskMolecularEncoder()
    elif model_type == 'contextpred':
        model = ContextPredMolecularEncoder()
    elif model_type == 'edgepred':
        model = EdgePredMolecularEncoder()
    elif model_type == 'moama':
        model = MoamaMolecularEncoder()
    else:
        raise ValueError("Invalid model type. Choose from 'attrmask', 'contextpred', 'edgepred', or 'moama'.")
    
    # Train the model
    model.fit(train_smiles_list)
    
    # Save the trained model
    model.save_to_local(os.path.join(output_dir, f"{model_type}_model.pth"))
    
    print(f"Trained {model_type} model saved to {output_dir}.")
    
    model.save_to_hf(
        repo_id=args.repo_id,
        hf_token=args.hf_token
    )
    
    print(f"Trained {model_type} model saved to Hugging Face Hub {args.repo_id}.")
    
def load_model(model_type='attrmask'):
    # Load the specified model
    if model_type == 'attrmask':
        model = AttrMaskMolecularEncoder()
        repo_id=args.repo_id
        file_name='AttrMaskMolecularEncoder.pt'
    elif model_type == 'contextpred':
        model = ContextPredMolecularEncoder()
        repo_id=repo_id=args.repo_id
        file_name='ContextPredMolecularEncoder.pt'
    elif model_type == 'edgepred':
        model = EdgePredMolecularEncoder()
        repo_id=repo_id=args.repo_id
        file_name='EdgePredMolecularEncoder.pt'
    elif model_type == 'moama':
        model = MoamaMolecularEncoder()
        repo_id=repo_id=args.repo_id
        file_name='MoamaMolecularEncoder.pt'
    else:
        raise ValueError("Invalid model type. Choose from 'attrmask', 'contextpred', 'edgepred', or 'moama'.")
    
    model.load_from_hf(repo_id=repo_id,
                       local_cache=file_name,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train molecular property prediction models')
    parser.add_argument('--model', type=str, choices=['attrmask', 'contextpred', 'edgepred', 'moama'], default='attrmask',
                        help='Type of model to train (default: attrmask)')
    parser.add_argument('--repo_id', type=str, default='',
                        help='Hugging Face Hub repo ID for the model')
    parser.add_argument('--hf_token', type=str, default='',
                        help='Hugging Face Hub token for authentication')
    args = parser.parse_args()
    
    train_model(model_type=args.model)
    
    load_model(model_type=args.model)