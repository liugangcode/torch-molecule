import os
import glob

# Find all files starting with test_run_ and ending with .py
files = glob.glob("run_*.py")

for old_name in files:
    # Replace 'test_run_' with 'run_' in the filename
    new_name = old_name.replace("test_run_", "run_", 1)
    os.rename(old_name, new_name)
    print(f"Renamed: {old_name} -> {new_name}")
