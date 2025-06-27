import os
import numpy as np
import gzip
import csv
from typing import List, Tuple

current_file_path = os.path.dirname(os.path.abspath(__file__))

def _load_from_local_csv(
    filename: str,
    input_cols: List[str],
    target_cols: List[str]
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Generic function to load data from local CSV.gz file within torch_molecule package.
    
    Args:
        filename (str): Name of the CSV.gz file in the data directory
        input_cols (List[str]): List of input column names (e.g., SMILES)
        target_cols (List[str]): List of target column names
    
    Returns:
        Tuple[List[List[str]], np.ndarray]: 
            - input_data: List of lists containing input data (e.g., SMILES strings)
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
    """
    data_path = os.path.join(current_file_path, "data", filename)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # print(f"Loading dataset from {data_path}")
    
    input_data = []
    property_data = []
    
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        
        # Check if required columns exist
        all_required_cols = input_cols + target_cols
        missing_cols = [col for col in all_required_cols if col not in csv_reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Required columns {missing_cols} not found in dataset. Available columns: {list(csv_reader.fieldnames)}")
        
        # Read data
        for row in csv_reader:
            # Get input data (e.g., SMILES)
            input_row = [row[col] for col in input_cols]
            input_data.append(input_row)
            
            # Get target data
            property_row = []
            for col in target_cols:
                try:
                    # Convert to float, handle empty strings or None
                    value = row[col]
                    if value == '' or value is None:
                        property_row.append(np.nan)
                    else:
                        property_row.append(float(value))
                except (ValueError, TypeError):
                    property_row.append(np.nan)
            property_data.append(property_row)
    
    property_numpy = np.array(property_data)
        
    return input_data, property_numpy

def load_gasperm(
    target_cols: List[str] = ["CH4", "CO2", "H2", "N2", "O2"],
) -> Tuple[List[str], np.ndarray]:
    """
    Load gas permeability dataset from local CSV.gz file within torch_molecule package.
    
    Args:
        target_cols (List[str]): List of target column names. Default is ["CH4", "CO2", "H2", "N2", "O2"]
    
    Returns:
        Tuple[List[str], np.ndarray]: 
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
    """
    input_cols = ["SMILES"]
    filename = "polymer_gas_permeability.csv.gz"
    
    input_data, property_numpy = _load_from_local_csv(filename, input_cols, target_cols)
    
    smiles_list = [row[0] for row in input_data]
    
    return smiles_list, property_numpy