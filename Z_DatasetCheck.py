# Z_DatasetCheck.py
# Author: QYH
# Version: 2.0
# Date: 2025/06/11
# This code is used for checking the dataset and defining:
# All the index can be modified in 'A_fiber_index.py'

import os
import numpy as np

if "__name__" == "__main__":
    # Define the dataset directory
    dataset_dir = "dataset_SNR_6.0dB"
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        exit(1)
    
    # List all files in the dataset directory
    files = os.listdir(dataset_dir)
    
    # Check if there are any files in the dataset
    if not files:
        print("The dataset directory is empty.")
        exit(1)
    
    # Check for specific file types (e.g., .npy files)
    npy_files = [f for f in files if f.endswith('.npy')]
    
    if not npy_files:
        print("No .npy files found in the dataset directory.")
        exit(1)
    
    print(f"Found {len(npy_files)} .npy files in the dataset directory.")
    
    # Optionally, load and check the shape of the first .npy file
    first_file_path = os.path.join(dataset_dir, npy_files[0])
    data = np.load(first_file_path)
    
    print(f"Shape of the first file '{npy_files[0]}': {data.shape}")