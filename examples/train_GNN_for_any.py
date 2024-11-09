import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
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
    if 'Polymer Type' in df.columns:
        df = df.drop(['Polymer Type'], axis=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    return df

def train_models(
    data_path: str = "gas_permeability.csv",
    hf_repo_id: Optional[str] = None,
    task: Optional[str] = None,
    n_trials: int = 100,
    model_type: str = "GNN",
):
    """
    Train GREA models for specified task(s).
    
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
    all_tasks = [col for col in df.columns if col not in ['SMILES', "PID" ,"Polymer Class"]]
    print('all_tasks', all_tasks)
    
    # Train a model for each selected gas
    for task in all_tasks:
        print(f"\n=== Training model for {data_path}  ===")
        print(f"Number of hyperparameter optimization trials: {n_trials}")
        
        # Prepare data for this task and filter out NaN values
        mask = df[task].notna()  # Remove rows where target is NaN
        X = df.loc[mask, 'SMILES'].values
        y = df.loc[mask, task].values
        
        # Additional check to remove any potential NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Print original data range
        print(f"Original data range for {task}:")
        print(f"Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")
        
        print(f"Number of valid samples for {task}: {len(y)}")
        
        y_train = y
        
        # Initialize model
        if model_type == 'GREA':
            print('Using GREA')
            model = GREAMolecularPredictor(
                num_tasks=1,
                task_type="regression",
                model_name=f"GREA_{task}",
                batch_size=256,
                epochs=100,
                evaluate_criterion='r2',
                evaluate_higher_better=True,
                verbose=True
            )
        elif model_type == 'GNN':
            print('Using GNN')
            model = GNNMolecularPredictor(
                num_tasks=1,
                task_type="regression",
                model_name=f"GNN_{task}",
                # loss_criterion=torch.nn.L1Loss(),
                augmented_feature=['maccs'], # maccs, morgan, or both
                batch_size=512,
                epochs=100,
                evaluate_criterion='r2',
                evaluate_higher_better=True,
                verbose=True
            )
        
        # Define search parameters
        search_parameters = {
            'gnn_type': ParameterSpec(
                ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]
            ),
            'norm_layer': ParameterSpec(
                ParameterType.CATEGORICAL,
                [
                    "batch_norm",
                    "layer_norm",
                    "size_norm",
                ],
            ),
            'augmented_feature': ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan", "maccs", "morgan", None]),
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
            # 'gamma': ParameterSpec(
            #     param_type=ParameterType.FLOAT,
            #     value_range=(0.25, 0.75)
            # )
        }
        
        print(f"\nTraining {task} model...")
        # Train model using all data for both training and validation
        model.autofit(
            X_train=X.tolist(),
            y_train=y_train,
            X_val=None,  # Use same data for validation
            y_val=None,     # Use same data for validation
            search_parameters=search_parameters,
            # n_trials=n_trials
            n_trials=2
        )
        # model.fit(
        #     X_train=X.tolist(),
        #     y_train=y_train,
        #     X_val=None,  # Use same data for validation
        #     y_val=None,     # Use same data for validation
        # )

        # Evaluate model on all data
        eval_results = model.predict(X.tolist())
        predictions = eval_results['prediction']
        print('predictions', predictions.max(), predictions.min())
        y_original = y
       
        metrics = {
            'mae': mean_absolute_error(y_original, predictions),
            'rmse': np.sqrt(mean_squared_error(y_original, predictions)),
            'r2': r2_score(y_original, predictions)
        }
        print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        # Upload to Hugging Face Hub
        if hf_repo_id:
            try:
                model.push_to_huggingface(
                    repo_id=hf_repo_id,
                    task_id=f"{task}",
                    metrics=metrics,
                    commit_message=f"Upload GREA_{task} model with metrics: {metrics}",
                    private=False
                )
                print(f"Successfully uploaded {task} model to Hugging Face Hub")
            except Exception as e:
                print(f"Failed to upload {task} model to Hugging Face Hub: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train task permeability models')
    parser.add_argument('--data_path', type=str, default="gas_permeability.csv",
                        help='Path to the gas permeability dataset')
    parser.add_argument('--hf_repo_id', type=Optional[str], default=None,
                        help='Hugging Face repository ID (optional)')
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--model_type', type=str, choices=['GNN', 'GREA'], default='GREA',
                        help='Type of model to train (GNN or GREA)')

    args = parser.parse_args()
    
    train_models(
        data_path=args.data_path,
        hf_repo_id=args.hf_repo_id,
        n_trials=args.n_trials,
        model_type=args.model_type,
    )
