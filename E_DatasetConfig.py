# E_DatasetConfig.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/26
# This code is used for loading the dual-channel dataset for training:
# curve dataset with dual channels and labels
# All the index can be modified in 'A_fiber_index.py'

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CurveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 包含数据的文件夹路径
        transform: 数据预处理操作
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        加载数据文件路径
        """
        samples = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith("_channel1.txt"):
                channel1_path = os.path.join(self.data_dir, file_name)
                channel2_path = channel1_path.replace("_channel1.txt", "_channel2.txt")
                label_path = channel1_path.replace("_channel1.txt", "_label.txt")
                samples.append((channel1_path, channel2_path, label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        channel1_path, channel2_path, label_path = self.samples[idx]

        # 加载数据
        channel1 = np.loadtxt(channel1_path)
        channel2 = np.loadtxt(channel2_path)
        label = np.loadtxt(label_path)

        # 转换为Tensor
        channel1 = torch.tensor(channel1, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        channel2 = torch.tensor(channel2, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            channel1 = self.transform(channel1)
            channel2 = self.transform(channel2)

        return channel1, channel2, label