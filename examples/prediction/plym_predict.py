import numpy as np
import pandas as pd
from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
import os
import shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import argparse
from torch_molecule.utils.search import ParameterType, ParameterSpec

search_GNN = {
    # Model architecture parameters
    "gnn_type": ParameterSpec(
        ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]
    ),
    "norm_layer": ParameterSpec(
        ParameterType.CATEGORICAL,
        [
            "batch_norm",
            "layer_norm",
        ],
    ),
    "graph_pooling": ParameterSpec(ParameterType.CATEGORICAL, ["mean", "sum", "max"]),
    "augmented_feature": ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan", "maccs", "morgan", None]),
    # Integer-valued parameters
    "num_layer": ParameterSpec(ParameterType.INTEGER, (2, 5)),
    "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 512)),
    "drop_ratio": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
    # Float-valued parameters with log scale
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-2)),
    "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-10, 1e-3)),
}

search_GREA = {
    "gamma": ParameterSpec(ParameterType.FLOAT, (0.25, 0.75)),
    **search_GNN
}

PATH = None # define the path to the dataset
N_trial = 200
N_epoch = 500
BATCH_SIZE = 128

def train_and_evaluate_models(model_type='both'):
    """
    Train multi-task models on the dataset and evaluate their performance.
    
    Args:
        model_type (str): Type of model to train - 'grea', 'gnn', or 'both'
    """
    # Load data
    train_data = pd.read_csv(f'{PATH}/train.csv')
    test_data = pd.read_csv(f'{PATH}/test.csv')
    
    # Get property columns (all columns except 'smiles')
    property_columns = train_data.columns.tolist()[1:]
    print(f"Property columns: {property_columns}")
    
    # Create output directory
    os.makedirs("./output_prediction", exist_ok=True)
    
    # Initialize results dataframes
    mae_results = pd.DataFrame(columns=property_columns)
    r2_results = pd.DataFrame(columns=property_columns)
    
    if model_type in ['grea', 'both']:
        grea_predictions = pd.DataFrame({'smiles': test_data['smiles']})
        mae_results.loc['GREA'] = np.nan
        r2_results.loc['GREA'] = np.nan
    
    if model_type in ['gnn', 'both']:
        gnn_predictions = pd.DataFrame({'smiles': test_data['smiles']})
        mae_results.loc['GNN'] = np.nan
        r2_results.loc['GNN'] = np.nan
    
    # Prepare training data for multi-task learning
    X = train_data['smiles']
    y = train_data[property_columns].values  # Shape [n_samples, n_tasks]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test data
    X_test = test_data['smiles']
    y_test = test_data[property_columns].values
    
    # Number of tasks is the number of properties we're predicting
    num_task = len(property_columns)
    
    # Train GREA model
    if model_type in ['grea', 'both']:
        print(f"Training GREA multi-task model for {num_task} properties...")
        grea_model = GREAMolecularPredictor(
            num_task=num_task,
            task_type="regression",
            model_name="GREA_multitask",
            batch_size=BATCH_SIZE,
            epochs=N_epoch,
            evaluate_criterion='r2',
            evaluate_higher_better=True,
            verbose=True
        )
        
        # Fit the model with hyperparameter optimization
        grea_model.autofit(
            X_train=X_train.tolist(),
            y_train=y_train,
            X_val=X_val.tolist(),
            y_val=y_val,
            n_trials=N_trial,
            search_parameters=search_GREA
        )
        
        # Save the model
        grea_model.save("./output_prediction/GREA_multitask.pt")
        
        # Predict on test data
        grea_output = grea_model.predict(test_data['smiles'].tolist())
        
        # Add predictions to dataframe
        for i, col in enumerate(property_columns):
            grea_predictions[col] = grea_output['prediction'][:, i]
            
            # Get non-null indices for this property
            non_null_idx = ~test_data[col].isna()
            
            if sum(non_null_idx) > 0:
                # Calculate MAE for non-null values
                grea_mae = mean_absolute_error(
                    test_data.loc[non_null_idx, col].values,
                    grea_output['prediction'][non_null_idx, i]
                )
                mae_results.loc['GREA', col] = grea_mae
                
                # Calculate R2 for non-null values
                grea_r2 = r2_score(
                    test_data.loc[non_null_idx, col].values,
                    grea_output['prediction'][non_null_idx, i]
                )
                r2_results.loc['GREA', col] = grea_r2
                
                print(f"GREA metrics for {col}: MAE = {grea_mae}, R² = {grea_r2}")
    
    # Train GNN model
    if model_type in ['gnn', 'both']:
        print(f"Training GNN multi-task model for {num_task} properties...")
        gnn_model = GNNMolecularPredictor(
            num_task=num_task,
            task_type="regression",
            model_name="GNN_multitask",
            batch_size=BATCH_SIZE,
            epochs=N_epoch,
            evaluate_criterion='r2',
            evaluate_higher_better=True,
            verbose=True
        )
        
        # Fit the model with hyperparameter optimization
        gnn_model.autofit(
            X_train=X_train.tolist(),
            y_train=y_train,
            X_val=X_val.tolist(),
            y_val=y_val,
            n_trials=N_trial,
            search_parameters=search_GNN
        )
        
        # Save the model
        gnn_model.save("./output_prediction/GNN_multitask.pt")
        
        # Predict on test data
        gnn_output = gnn_model.predict(test_data['smiles'].tolist())
        
        # Add predictions to dataframe
        for i, col in enumerate(property_columns):
            gnn_predictions[col] = gnn_output['prediction'][:, i]
            
            # Get non-null indices for this property
            non_null_idx = ~test_data[col].isna()
            
            if sum(non_null_idx) > 0:
                # Calculate MAE for non-null values
                gnn_mae = mean_absolute_error(
                    test_data.loc[non_null_idx, col].values,
                    gnn_output['prediction'][non_null_idx, i]
                )
                mae_results.loc['GNN', col] = gnn_mae
                
                # Calculate R2 for non-null values
                gnn_r2 = r2_score(
                    test_data.loc[non_null_idx, col].values,
                    gnn_output['prediction'][non_null_idx, i]
                )
                r2_results.loc['GNN', col] = gnn_r2
                
                print(f"GNN metrics for {col}: MAE = {gnn_mae}, R² = {gnn_r2}")
    
    # Save predictions and evaluation results
    if model_type in ['grea', 'both']:
        grea_predictions.to_csv("./output_prediction/predict_grea.csv", index=False)
    if model_type in ['gnn', 'both']:
        gnn_predictions.to_csv("./output_prediction/predict_gnn.csv", index=False)
    
    mae_results.to_csv("./output_prediction/mae_results.csv")
    r2_results.to_csv("./output_prediction/r2_results.csv")
    
    print("\nTraining and evaluation completed successfully.")
    print(f"MAE results saved to ./output_prediction/mae_results.csv")
    print(f"R² results saved to ./output_prediction/r2_results.csv")
    if model_type in ['grea', 'both']:
        print(f"GREA predictions saved to ./output_prediction/predict_grea.csv")
    if model_type in ['gnn', 'both']:
        print(f"GNN predictions saved to ./output_prediction/predict_gnn.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train molecular property prediction models')
    parser.add_argument('--model', type=str, choices=['grea', 'gnn', 'both'], default='both',
                        help='Type of model to train (grea, gnn, or both)')
    args = parser.parse_args()
    
    train_and_evaluate_models(model_type=args.model) 