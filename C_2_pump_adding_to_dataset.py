# C_2_pump_adding_to_dataset.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/26
# This code is used for adding pump distribution dataset_clean to the existing dataset_clean of network:
# Generation of pump distribution dataset_clean
# All the index can be modified in 'A_fiber_index.py'

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define the path to the existing data

    ## pump data
    pump_path = "pump_data/distorted_pump_under_0.01-5_mW.npy"
    ## dataset_clean data
    test_dataset_path = "dataset_clean/Dataset_Pumppower_index_0_SNR_Clean.npy"
    ## label data
    test_label_path = "dataset_clean/Label_Pumppower_index_0_SNR_Clean.npy"
    # load data
    pump_data = np.load(pump_path)
    dataset_data = np.load(test_dataset_path)
    label_data = np.load(test_label_path)
    # Print the shapes of the loaded data
    print("Pump data shape:", pump_data.shape)
    print("Dataset data shape:", dataset_data.shape)
    print("Label data shape:", label_data.shape)
    # Create a new array for pump set with the same shape as dataset_data
    pump_set = np.zeros_like((dataset_data), dtype=np.float32)
    ## check the shape of pump_set
    print("Pump set shape:", pump_set.shape)
    ## Fill the pump_set with the pump_data
    for i in range(pump_set.shape[1]):
        pump_set[:, i] = pump_data[:, 0] 
        ### All the same pump
    # Save the pump_set to a new file
    np.save("dataset_clean/Pump_Power_index_0.npy", pump_set)