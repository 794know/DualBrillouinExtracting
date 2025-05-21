# D_noise_add.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/21
# This code is used for adding noise to the simulated clean dataset:
# Generation of dual-peaks BGS dataset with noise
# All the index can be modified in 'A_fiber_index.py'

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # 指定文件夹路径
    folder_path = 'dataset_clean'

    # 构建完整的文件路径
    dataset_path = os.path.join(folder_path, 'Dataset_Pumppower_index_0_SNR_Clean.npy')
    label_path = os.path.join(folder_path, 'Label_Pumppower_index_0_SNR_Clean.npy')

    # 加载数据
    dataset_clean = np.load(dataset_path)
    label_clean = np.load(label_path)

    # 检查数据形状
    print("Dataset shape:", dataset_clean.shape)
    print("Label shape:", label_clean.shape)

    
    # 选择固定 X 值（例如 X=10，可以根据需要调整）
    fixed_x = 132299  # 选择第 10 列（索引从 0 开始）

    # 提取固定 X 值处的数据
    dataset_yz = dataset_clean[:, fixed_x]
    label_yz = label_clean[:, fixed_x]

    # 绘制 Dataset_Pumppower_index_0_SNR_Clean.npy 的 Y-Z 图
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_yz, label=f'Dataset (X={fixed_x})')
    plt.title('Dataset_Pumppower_index_0_SNR_Clean Y-Z Plane')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 输出 Label_Pumppower_index_0_SNR_Clean.npy 在该 X 值处的两个值
    print(f"Label values at X={fixed_x}:")
    print("Label 1:", label_yz[0])
    print("Label 2:", label_yz[1])

    # 绘制 Label_Pumppower_index_0_SNR_Clean.npy 的两个值
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], label_yz[:2], 'ro-', label=f'Label (X={fixed_x})')
    plt.title('Label_Pumppower_index_0_SNR_Clean Values')
    plt.xlabel('Index')
    plt.ylabel('Label Value')
    plt.xticks([0, 1], ['Label 1', 'Label 2'])
    plt.legend()
    plt.grid(True)
    plt.show()
    