from torch_molecule.datasets import load_gasperm
import numpy as np


def test_gasperm_download_and_cleanup():
    """
    Test gas permeability dataset loading, print results, and cleanup local files.
    """
    print("=" * 60)
    print("Testing Gas Permeability Dataset Loading")
    print("=" * 60)
    
    try:
        print(f"\n1. Testing loading with default target columns")
        print("-" * 40)
        
        # Test with default target columns
        molecular_dataset = load_gasperm()
        smiles_list = molecular_dataset.data
        property_numpy = molecular_dataset.target
        
        # Print results
        print(f"\nResults:")
        print(f"- Number of molecules: {len(smiles_list)}")
        print(f"- Property array shape: {property_numpy.shape}")
        print(f"- Target columns: ['CH4', 'CO2', 'H2', 'N2', 'O2']")
        
        print(f"\nFirst 5 SMILES:")
        for i, smiles in enumerate(smiles_list[:5]):
            print(f"  {i+1}. {smiles}")
        
        print(f"\nFirst 5 property values (all gases):")
        for i, prop in enumerate(property_numpy[:5]):
            print(f"  {i+1}. {prop}")
        
        print(f"\nProperty statistics for each gas:")
        gas_names = ['CH4', 'CO2', 'H2', 'N2', 'O2']
        for j, gas in enumerate(gas_names):
            gas_values = property_numpy[:, j]
            # Filter out NaN values for statistics
            valid_values = gas_values[~np.isnan(gas_values)]
            if len(valid_values) > 0:
                print(f"  {gas}:")
                print(f"    Min: {valid_values.min():.6f}")
                print(f"    Max: {valid_values.max():.6f}")
                print(f"    Mean: {valid_values.mean():.6f}")
                print(f"    Std: {valid_values.std():.6f}")
                print(f"    Valid values: {len(valid_values)}/{len(gas_values)}")
            else:
                print(f"  {gas}: No valid values found")
        
        # Test with custom target columns
        print(f"\n2. Testing with custom target columns")
        print("-" * 40)
        
        custom_targets = ["CH4", "CO2"]
        molecular_dataset2 = load_gasperm(target_cols=custom_targets)
        smiles_list2 = molecular_dataset2.data
        property_numpy2 = molecular_dataset2.target
        
        print(f"Custom target results:")
        print(f"- Target columns: {custom_targets}")
        print(f"- Property array shape: {property_numpy2.shape}")
        print(f"- Same number of molecules: {len(smiles_list2) == len(smiles_list)}")
        print(f"- First molecule properties: {property_numpy2[0]}")
        
        # Test with single target column
        print(f"\n3. Testing with single target column")
        print("-" * 40)
        
        single_target = ["H2"]
        molecular_dataset3 = load_gasperm(target_cols=single_target)
        smiles_list3 = molecular_dataset3.data
        property_numpy3 = molecular_dataset3.target
        
        print(f"Single target results:")
        print(f"- Target columns: {single_target}")
        print(f"- Property array shape: {property_numpy3.shape}")
        print(f"- First 5 H2 permeability values:")
        for i, prop in enumerate(property_numpy3[:5]):
            print(f"  {i+1}. {prop[0]:.6f}" if not np.isnan(prop[0]) else f"  {i+1}. NaN")
        
        # Test error handling with invalid target column
        print(f"\n4. Testing error handling with invalid target column")
        print("-" * 40)
        
        try:
            invalid_targets = ["INVALID_GAS"]
            molecular_dataset4 = load_gasperm(target_cols=invalid_targets)
            smiles_list4 = molecular_dataset4.data
            property_numpy4 = molecular_dataset4.target
            print("ERROR: Should have raised ValueError for invalid target column")
        except ValueError as e:
            print(f"Successfully caught expected error: {e}")
        except Exception as e:
            print(f"Unexpected error type: {type(e).__name__}: {e}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("Gas Permeability Dataset Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_gasperm_download_and_cleanup()