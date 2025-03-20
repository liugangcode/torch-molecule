import numpy as np
import pandas as pd
from torch_molecule import GREAMolecularPredictor
import os
import shutil
from pathlib import Path

def predict_gas_permeability(
    data_path: str,
    repo_id: str = "liuganghuggingface/torch-molecule-ckpt-GREA-gas-separation",
    log_scale: bool = False
):
    """
    Predict gas permeability using downloaded models and clean up afterwards.
    
    Args:
        data_path: Path to input CSV file containing SMILES strings
        repo_id: Hugging Face repository ID
        log_scale: Whether models were trained in log scale
    """
    try:
        # Load input data
        df = pd.read_csv(data_path)
        smiles_list = df['SMILES'].tolist()
        results_df = pd.DataFrame({'SMILES': df['SMILES']})
        
        # Define gases
        gases = ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']
        
        # Store predictions temporarily
        original_preds = {}
        log_preds = {}
        
        if log_scale:
            repo_id = repo_id + '-logscale'
            
        # Create temporary directory for downloaded models
        model_dir = "./downloaded_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Make predictions for each gas
        for gas in gases:
            print(f"\nPredicting {gas} permeability...")
            
            try:
                # Load model
                model = GREAMolecularPredictor()
                model.load_model(f"{model_dir}/GREA_{gas}.pt", repo_id=repo_id)
                model.set_params(verbose=True)
                # Make predictions
                predictions = model.predict(smiles_list)
                pred_values = predictions['prediction']
                pred_variance = predictions['variance'].reshape(-1)
                node_importance = predictions['node_importance']
                print('top-10 example for prediction variance', pred_variance[:10])
                print('top-1 example for node importance', node_importance[0])

                # Convert predictions if needed
                if log_scale:
                    original_scale_pred = 10**pred_values
                    log_scale_pred = pred_values
                else:
                    original_scale_pred = np.maximum(pred_values, 1e-8)
                    log_scale_pred = np.log10(original_scale_pred)
                
                # Store predictions
                original_preds[gas] = original_scale_pred
                log_preds[gas] = log_scale_pred
                
            except Exception as e:
                print(f"Failed to predict {gas} permeability: {str(e)}")
                original_preds[gas] = np.full(len(smiles_list), np.nan)
                log_preds[gas] = np.full(len(smiles_list), np.nan)
        
        # Add predictions to DataFrame in desired order
        # First: original scale predictions
        for gas in gases:
            results_df[f'{gas}'] = original_preds[gas]
        
        # Second: log scale predictions
        for gas in gases:
            results_df[f'{gas}_log'] = log_preds[gas]
                
        # Save predictions
        output_path = str(Path(data_path).parent / f"{Path(data_path).stem}_predicted.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved predictions to: {output_path}")
        print("\nColumn order in output file:")
        print("1. SMILES")
        print("2. Original scale predictions:", ", ".join(gases))
        print("3. Log scale predictions:", ", ".join([f"{gas}_log" for gas in gases]))
        
    except Exception as e:
        print(f"Prediction failed with error: {str(e)}")
        print("Note: Make sure HF_TOKEN environment variable is set")
        
    finally:
        # Clean up downloaded files
        if os.path.exists("./downloaded_model"):
            shutil.rmtree("./downloaded_model")
            print("Cleaned up downloaded model files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict gas permeability')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file containing SMILES strings')
    parser.add_argument('--log_scale', action='store_true',
                        help='Whether models were trained in log scale')
    
    args = parser.parse_args()
    
    predict_gas_permeability(
        data_path=args.data_path,
        log_scale=args.log_scale
    )