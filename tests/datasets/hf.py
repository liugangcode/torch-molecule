import os
import tempfile
import shutil
from torch_molecule.datasets import load_qm9, load_chembl2k, load_broad6k, load_toxcast, load_admet
import numpy as np
import csv
import gzip

def load_dataset(dataset_name="qm9"):
    if dataset_name == "qm9":
        return load_qm9
    elif dataset_name == "chembl2k":
        return load_chembl2k
    elif dataset_name == "broad6k":
        return load_broad6k
    elif dataset_name == "toxcast":
        return load_toxcast
    elif dataset_name == "admet":
        return load_admet
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def test_download_and_cleanup(dataset_name="qm9"):
    """
    Test QM9 dataset download, print results, and cleanup local files.
    """
    print("=" * 60)
    print(f"Testing {dataset_name} Dataset Download and Loading")
    print("=" * 60)
    
    # Create a temporary directory for testing
    if False:
        temp_dir = tempfile.mkdtemp()
        test_csv_path = temp_dir
    else:
        test_csv_path = "torchmol_data"

    load_func = load_dataset(dataset_name)
    
    try:
        print(f"\n1. Testing download to temporary location: {test_csv_path}")
        print("-" * 40)
        
        # Test with default target columns
        smiles_list, property_numpy, local_data_path = load_func(
            local_dir=test_csv_path,
            return_local_data_path=True,
        )
        
        # Print results
        print(f"\nResults:")
        print(f"- Number of molecules: {len(smiles_list)}")
        print(f"- Property array shape: {property_numpy.shape}")
        print(f"- File exists: {os.path.exists(local_data_path)}")
        print(f"- File size: {os.path.getsize(local_data_path) if os.path.exists(local_data_path) else 0} bytes")
        
        print(f"\nFirst 5 SMILES:")
        for i, smiles in enumerate(smiles_list[:5]):
            print(f"  {i+1}. {smiles}")
        
        print(f"\nFirst 5 property values (gap):")
        for i, prop in enumerate(property_numpy[:5]):
            print(f"  {i+1}. {prop[0]:.6f}")
        
        print(f"\nProperty statistics:")
        # Calculate statistics excluding NaN values
        non_null_mask = ~np.isnan(property_numpy)
        non_null_values = property_numpy[non_null_mask]
        
        print(f"  Total values: {property_numpy.size}")
        print(f"  Non-null values: {non_null_values.size}")
        print(f"  Null values: {property_numpy.size - non_null_values.size}")
        print(f"  Non-null percentage: {(non_null_values.size / property_numpy.size * 100):.2f}%")
        
        if non_null_values.size > 0:
            print(f"  Min (non-null): {non_null_values.min():.6f}")
            print(f"  Max (non-null): {non_null_values.max():.6f}")
            print(f"  Mean (non-null): {non_null_values.mean():.6f}")
            print(f"  Std (non-null): {non_null_values.std():.6f}")
        else:
            print("  No non-null values found")
        
        # Test loading from existing file (should not download again)
        print(f"\n2. Testing loading from existing file")
        print("-" * 40)
        
        smiles_list2, property_numpy2, local_data_path = load_func(
            local_dir=test_csv_path,
            return_local_data_path=True,
        )
        
        print(f"Second load results:")
        print(f"- Same number of molecules: {len(smiles_list2) == len(smiles_list)}")
        print(f"- Same property shape: {property_numpy2.shape == property_numpy.shape}")
        print(f"- Local data path: {local_data_path}")
        
        # Test with multiple target columns (if available)
        print(f"\n3. Testing with multiple target columns")
        print("-" * 40)
        # check if .gz file
        if local_data_path.endswith('.gz'):
            with gzip.open(local_data_path, 'rt', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                available_cols = list(reader.fieldnames) # type: ignore
            print(f"Available columns: {available_cols}")
        else:
            with open(local_data_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                available_cols = list(reader.fieldnames) # type: ignore
            print(f"Available columns: {available_cols}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    
    finally:
        # Cleanup: Delete the temporary directory and all files
        print(f"\n4. Cleaning up temporary files")
        print("-" * 40)
        
        if os.path.exists(test_csv_path):
            file_count = len(os.listdir(test_csv_path))
            shutil.rmtree(test_csv_path)
            print(f"- Deleted temporary directory: {test_csv_path}")
            print(f"- Removed {file_count} file(s)")
        
        print(f"- Cleanup completed successfully")
    
    print("\n" + "=" * 60)
    print(f"{dataset_name} Dataset Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_download_and_cleanup(dataset_name="qm9")
    test_download_and_cleanup(dataset_name="chembl2k")
    test_download_and_cleanup(dataset_name="broad6k")
    test_download_and_cleanup(dataset_name="toxcast")
    test_download_and_cleanup(dataset_name="admet")
