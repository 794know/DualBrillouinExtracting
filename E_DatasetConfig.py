# E_DatasetConfig.py
# Author: QYH
# Version: 2.0
# Date: 2025/06/12
# This code is used for loading the DualpeakBGS for training:
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
    # 每一行的最大值
    max_values = np.max(data, axis=1, keepdims=True)
    max_values[max_values == 0] = 1
    normalized_data = data / max_values
    return normalized_data


def normalize_labels(labels):
    """
    对标签数据进行归一化，使其除以该label列的最大值。
    如果数据的最大值为0，则保持原样（避免除以0）。
    """
    # 每一列的最大值
    max_values = np.max(labels, axis=0, keepdims=True)
    max_values[max_values == 0] = 1
    normalized_labels = labels / max_values
    return normalized_labels


class DataDBGS(Dataset):
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
                # Noisy Data
                clean_path = os.path.join(data_dir, f"Dataset_Pumppower_index_0_SNR_{os.path.basename(data_dir).split('_')[-1]}.npy")
            else:
                # Clean Data
                clean_path = os.path.join(data_dir, "Dataset_Pumppower_index_0_SNR_Clean.npy")
            # Label Path
            label_path = os.path.join(data_dir, "Label_Pumppower_index_0_SNR_Clean.npy")
            # 加载数据
            clean_data = np.load(clean_path).T  # 转置为 [132300, 600]
            labels = np.load(label_path).T  # 转置为 [132300, 2]
            
            # 对每个数据进行归一化
            clean_data = normalize_data(clean_data)
            labels = normalize_labels(labels)

            # 确保数据长度一致
            min_length = min(len(clean_data), len(labels))
            clean_data = clean_data[:min_length]
            labels = labels[:min_length]
            
            # 将数据添加到样本列表
            for i in range(min_length):
                samples.append((clean_data[i], labels[i]))
        
        return samples
    
    def _print_labels(self):
        """
        打印前几个 label 样本
        """
        print("First few label samples:")
        for i in range(min(10, len(self.samples))):  # 打印前10个样本
            print(f"Sample {i+1}: {self.samples[i][1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean, label = self.samples[idx]

        # 转换为Tensor
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        label = torch.tensor(label, dtype=torch.float32)

        return clean, label

if __name__ == "__main__":
    
    # 测试代码
    import matplotlib.pyplot as plt
    # 测试DataDBGS数据集加载
    data_dirs = [
        "dataset_clean",
        "dataset_SNR_12.0dB",
        "dataset_SNR_15.0dB"
    ]
    dataset = DataDBGS(data_dirs)
    print(f"Total samples: {len(dataset)}")
    # 打印前10个样本的标签
    print("First 10 labels:")
    for i in range(10):
        _, label = dataset[i]
        print(f"Label {i+1}: {label.numpy()}")

    plt.figure(figsize=(12, 6))
    for i in range(132000, 132500):
        clean, label = dataset[i]
        clean_np = clean.squeeze(0).numpy()  # [600]
        plt.plot(clean_np, label=f"Sample {i+1} | Label: {label.numpy()}")

    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Value")
    plt.title("Samples from 132000 to 132300 and Their Labels")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()