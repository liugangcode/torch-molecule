import os
import csv
import numpy as np
from tqdm import tqdm

import torch
from torch_molecule.generator.lstm import LSTMMolecularGenerator

# EPOCHS = 1000  # Reduced for faster testing
EPOCHS = 5
BATCH_SIZE = 24

def test_lstm_generator():
    # Load data from polymer100.csv
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "data", "polymer100.csv")
    print(f"Loading data from: {data_path}")
    
    # Read CSV without pandas
    smiles_list = []
    properties = []
    property_columns = []
    
    with open(data_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Get property columns (all columns except 'smiles')
        property_columns = [col for col in reader.fieldnames if col != 'smiles']
        
        for row in reader:
            smiles_list.append(row['smiles'])
            # Extract property values for this row
            row_properties = [float(row[col]) for col in property_columns]
            properties.append(row_properties)
    
    print(f"Loaded {len(smiles_list)} molecules with {len(property_columns)} properties")
    print(f"Property columns: {property_columns}")
    print(f"First 3 SMILES: {smiles_list[:3]}")
    print(f"First 3 properties: {properties[:3]}")
    
    # 1. Basic initialization test - Unconditional Model
    print("\n=== Testing Unconditional LSTM model initialization ===")
    unconditional_model = LSTMMolecularGenerator(
        num_layer=3,
        hidden_size=128,
        max_len=64,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True
    )
    print("Unconditional LSTM Model initialized successfully")

    # 2. Basic fitting test - Unconditional Model
    print("\n=== Testing Unconditional LSTM model fitting ===")
    unconditional_model.fit(smiles_list)
    print("Unconditional LSTM Model fitting completed")

    # 3. Unconditional generation test
    print("\n=== Testing Unconditional LSTM generation ===")
    generated_smiles_uncond = unconditional_model.generate(batch_size=BATCH_SIZE)
    print(f"Unconditionally generated {len(generated_smiles_uncond)} molecules")
    print("Example unconditionally generated SMILES:", generated_smiles_uncond[:10])
    
    # 4. Model saving and loading test - Unconditional Model
    print("\n=== Testing Unconditional LSTM model saving and loading ===")
    save_path = "unconditional_lstm_test_model.pt"
    unconditional_model.save_to_local(save_path)
    print(f"Unconditional LSTM Model saved to {save_path}")

    new_unconditional_model = LSTMMolecularGenerator()
    new_unconditional_model.load_from_local(save_path)
    print("Unconditional LSTM Model loaded successfully")

    # Test generation with loaded unconditional model
    generated_smiles_uncond = new_unconditional_model.generate(batch_size=5)
    print("Generated molecules with loaded unconditional model:", len(generated_smiles_uncond))
    print("Example generated SMILES:", generated_smiles_uncond[:10])

    # Clean up unconditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")
    
    # 5. Basic initialization test - Property Conditional Model
    print("\n=== Testing Property Conditional LSTM model initialization ===")
    prop_conditional_model = LSTMMolecularGenerator(
        num_layer=2,
        hidden_size=128,
        max_len=64,
        num_task=len(property_columns),  # Set number of properties
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True
    )
    print("Property Conditional LSTM Model initialized successfully")

    # 6. Basic fitting test - Property Conditional Model
    print("\n=== Testing Property Conditional LSTM model fitting ===")
    prop_conditional_model.fit(smiles_list, properties)
    print("Property Conditional LSTM Model fitting completed")

    # 7. Property conditional generation test
    print("\n=== Testing Property Conditional LSTM generation ===")
    # Create some target properties (using mean values from the dataset as a starting point)
    mean_properties = np.mean(properties, axis=0).tolist()
    target_properties = []
    for i in range(5):
        # Create variations around the mean
        target_prop = [p * (0.8 + 0.4 * np.random.random()) for p in mean_properties]
        target_properties.append(target_prop)
    
    print(f"Target properties for generation: {target_properties}")
    generated_smiles = prop_conditional_model.generate(labels=target_properties)
    print(f"Property conditionally generated {len(generated_smiles)} molecules")
    print("Example property conditionally generated SMILES:", generated_smiles[:2])
    
    # 8. Model saving and loading test - Property Conditional Model
    print("\n=== Testing Property Conditional LSTM model saving and loading ===")
    save_path = "prop_conditional_lstm_test_model.pt"
    prop_conditional_model.save_to_local(save_path)
    print(f"Property Conditional LSTM Model saved to {save_path}")

    new_prop_conditional_model = LSTMMolecularGenerator()
    new_prop_conditional_model.load_from_local(save_path)
    print("Property Conditional LSTM Model loaded successfully")

    # Test generation with loaded property conditional model
    generated_smiles = new_prop_conditional_model.generate(labels=target_properties)
    print("Generated molecules with loaded property conditional model:", len(generated_smiles))
    print("Example generated SMILES:", generated_smiles[:2])

    # Clean up property conditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_lstm_generator()