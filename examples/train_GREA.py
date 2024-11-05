import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from torch_molecule import GREAMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
from torch_molecule.utils import mean_absolute_error, mean_squared_error, r2_score
from torch_molecule.utils.search import ParameterType, ParameterSpec

def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Preprocess the gas permeability dataset.
    """
    # Read and clean data
    df = pd.read_csv(data_path)
    # Remove unnamed index column and Polymer Type column
    df = df.drop(['Polymer Type'], axis=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    return df

def train_gas_permeability_models(
    data_path: str = "gas_permeability.csv",
    hf_repo_id: str = "liuganghuggingface/torch-molecule-ckpt-GREA-gas-separation",
    task: Optional[str] = None,
    n_trials: int = 100,
    train_in_log: bool = True
):
    """
    Train GREA models for specified gas permeability task(s).
    
    Args:
        data_path: Path to gas permeability CSV file
        hf_repo_id: Hugging Face repository ID for uploading
        task: Specific gas to train for (He, H2, O2, N2, CO2, CH4). If None, trains for all gases.
        n_trials: Number of trials for hyperparameter optimization (default: 100)
        train_in_log: Whether to train the model in log space (default: True)
    """
    # Load and preprocess data
    df = preprocess_data(data_path)
    
    # Get list of gas tasks (all columns except SMILES)
    all_gas_tasks = [col for col in df.columns if col != 'SMILES']
    
    # Validate task argument if provided
    if task is not None:
        if task not in all_gas_tasks:
            raise ValueError(f"Invalid task: {task}. Must be one of {all_gas_tasks}")
        gas_tasks = [task]
    else:
        gas_tasks = all_gas_tasks
    
    # Train a model for each selected gas
    for gas in gas_tasks:
        print(f"\n=== Training model for {gas} permeability ===")
        print(f"Number of hyperparameter optimization trials: {n_trials}")
        print(f"Training in {'log' if train_in_log else 'original'} space")
        
        # Prepare data for this gas and filter out NaN values
        mask = df[gas].notna()  # Remove rows where target is NaN
        X = df.loc[mask, 'SMILES'].values
        y = df.loc[mask, gas].values
        
        # Additional check to remove any potential NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Print original data range
        print(f"Original data range for {gas}:")
        print(f"Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")
        
        print(f"Number of valid samples for {gas}: {len(y)}")
        
        # Convert to log scale if specified
        if train_in_log:
            y_train = np.log10(y)
            print(f"Log-transformed data range for {gas}:")
            print(f"Min: {np.min(y_train):.4f}, Max: {np.max(y_train):.4f}")
        else:
            y_train = y
        
        # Initialize model
        model = GREAMolecularPredictor(
            num_tasks=1,
            task_type="regression",
            model_name=f"GREA_{gas}",
            batch_size=512,
            epochs=500,
            evaluate_criterion='r2',
            evaluate_higher_better=True,
            verbose=True
        )
        
        # Define search parameters
        search_parameters = {
            "gnn_type": ParameterSpec(
                ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]
            ),
            "norm_layer": ParameterSpec(
                ParameterType.CATEGORICAL,
                [
                    "batch_norm",
                    "layer_norm",
                    "size_norm",
                ],
            ),
            'num_layer': ParameterSpec(
                param_type=ParameterType.INTEGER,
                value_range=(2, 5)
            ),
            'emb_dim': ParameterSpec(
                param_type=ParameterType.INTEGER,
                value_range=(256, 512)
            ),
            'learning_rate': ParameterSpec(
                param_type=ParameterType.FLOAT,
                value_range=(1e-4, 1e-2)
            ),
            'drop_ratio': ParameterSpec(
                param_type=ParameterType.FLOAT,
                value_range=(0.05, 0.5)
            ),
            'gamma': ParameterSpec(
                param_type=ParameterType.FLOAT,
                value_range=(0.25, 0.75)
            )
        }
        
        print(f"\nTraining {gas} model...")
        # Train model using all data for both training and validation
        model.autofit(
            X_train=X.tolist(),
            y_train=y_train,
            X_val=X.tolist(),  # Use same data for validation
            y_val=y_train,     # Use same data for validation
            search_parameters=search_parameters,
            n_trials=n_trials
        )
        
        # Evaluate model on all data
        eval_results = model.predict(X.tolist())
        predictions = eval_results['prediction']
        
        if train_in_log:
            # Calculate metrics in log scale
            metrics = {
                'mae_logscale': mean_absolute_error(y_train, predictions),
                'rmse_logscale': np.sqrt(mean_squared_error(y_train, predictions)),
                'r2_logscale': r2_score(y_train, predictions)
            }
            print(f"\n{gas} Model Performance:")
            print(f"Log Scale - MAE: {metrics['mae_logscale']:.4f}, RMSE: {metrics['rmse_logscale']:.4f}, R²: {metrics['r2_logscale']:.4f}")

            pred_original = 10**predictions
            y_original = y
        else:
            # For models trained in original space
            pred_original = np.maximum(predictions, 1e-8)
            y_original = y
            
            # Calculate log-scale metrics by transforming to log space
            y_log = np.log10(y)
            pred_log = np.log10(pred_original)  # Safe now because of the minimum threshold
            metrics = {
                'mae_logscale': mean_absolute_error(y_log, pred_log),
                'rmse_logscale': np.sqrt(mean_squared_error(y_log, pred_log)),
                'r2_logscale': r2_score(y_log, pred_log)
            }
            print(f"\n{gas} Model Performance:")
            print(f"Log Scale - MAE: {metrics['mae_logscale']:.4f}, RMSE: {metrics['rmse_logscale']:.4f}, R²: {metrics['r2_logscale']:.4f}")

        # Calculate metrics in original scale
        metrics['mae_original'] = mean_absolute_error(y_original, pred_original)
        metrics['rmse_original'] = np.sqrt(mean_squared_error(y_original, pred_original))
        metrics['r2_original'] = r2_score(y_original, pred_original)
        
        print(f"Original Scale - MAE: {metrics['mae_original']:.4f}, RMSE: {metrics['rmse_original']:.4f}, R²: {metrics['r2_original']:.4f}")
        
        # Upload to Hugging Face Hub
        try:
            model.push_to_huggingface(
                repo_id=hf_repo_id,
                task_id=f"{gas}",
                metrics=metrics,
                commit_message=f"Upload GREA_{gas} model with metrics: {metrics}",
                private=False
            )
            print(f"Successfully uploaded {gas} model to Hugging Face Hub")
        except Exception as e:
            print(f"Failed to upload {gas} model to Hugging Face Hub: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train gas permeability models')
    parser.add_argument('--data_path', type=str, default="gas_permeability.csv",
                        help='Path to the gas permeability dataset')
    parser.add_argument('--hf_repo_id', type=str, 
                        default="liuganghuggingface/torch-molecule-ckpt-GREA-gas-separation",
                        help='Hugging Face repository ID')
    parser.add_argument('--task', type=str, choices=['He', 'H2', 'O2', 'N2', 'CO2', 'CH4'],
                        help='Specific gas to train for. If not specified, trains for all gases.')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--train_in_log', action='store_true',
                        help='Whether to train the model in log space (default: False)')

    args = parser.parse_args()

    if args.train_in_log:
        args.hf_repo_id = args.hf_repo_id + '-logscale'
    
    train_gas_permeability_models(
        data_path=args.data_path,
        hf_repo_id=args.hf_repo_id,
        task=args.task,
        n_trials=args.n_trials,
        train_in_log=args.train_in_log
    )