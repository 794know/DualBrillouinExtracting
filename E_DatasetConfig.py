# E_DatasetConfig.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/26
# This code is used for loading the dual-channel dataset for training:
# curve dataset with dual channels and labels
# All the index can be modified in 'A_fiber_index.py'

import os
import torch
from torch.utils.data import Dataset
import numpy as np

# Normalize the data to ensure each sample is scaled correctly
def normalize_data(data):
    """
    对每个样本数据进行归一化，使其除以自身的最大值。
    如果数据的最大值为0，则保持原样（避免除以0）。
    """
    max_values = np.max(data, axis=1, keepdims=True)
    max_values[max_values == 0] = 1
    normalized_data = data / max_values
    return normalized_data

class PumpPowerDataset(Dataset):
    def __init__(self, data_dirs):
        """
        data_dirs: 包含多个数据文件夹路径的列表
        """
        self.data_dirs = data_dirs
        self.samples = self._load_samples()
        self._print_labels()  # 打印前几个 label 样本

    def _load_samples(self):
        """
        从多个文件夹中加载数据文件路径
        """
        samples = []
        for data_dir in self.data_dirs:
            # 动态选择文件名
            if "SNR" in os.path.basename(data_dir):
                clean_path = os.path.join(data_dir, f"Dataset_Pumppower_index_0_SNR_{os.path.basename(data_dir).split('_')[-1]}.npy")
            else:
                clean_path = os.path.join(data_dir, "Dataset_Pumppower_index_0_SNR_Clean.npy")
            
            distorted_path = os.path.join(data_dir, "Pump_Power_index_0.npy")
            label_path = os.path.join(data_dir, "Label_Pumppower_index_0_SNR_Clean.npy")
            
            # 检查文件是否存在
            if not os.path.exists(clean_path) or not os.path.exists(distorted_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"文件缺失: {clean_path}, {distorted_path}, {label_path}")
            
            # 加载数据
            clean_data = np.load(clean_path).T  # 转置为 [132300, 600]
            distorted_data = np.load(distorted_path).T  # 转置为 [132300, 600]
            labels = np.load(label_path).T  # 转置为 [132300, 2]
            
            # 对每个数据进行归一化
            clean_data = normalize_data(clean_data)
            distorted_data = normalize_data(distorted_data)
            labels = normalize_data(labels)

            # 确保数据长度一致
            min_length = min(len(clean_data), len(distorted_data), len(labels))
            clean_data = clean_data[:min_length]
            distorted_data = distorted_data[:min_length]
            labels = labels[:min_length]
            
            # 将数据添加到样本列表
            for i in range(min_length):
                samples.append((clean_data[i], distorted_data[i], labels[i]))
        
        return samples
    
    def _print_labels(self):
        """
        打印前几个 label 样本
        """
        print("First few label samples:")
        for i in range(min(10, len(self.samples))):  # 打印前10个样本
            print(f"Sample {i+1}: {self.samples[i][2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean, distorted, label = self.samples[idx]

        # 转换为Tensor
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        distorted = torch.tensor(distorted, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return clean, distorted, label
    