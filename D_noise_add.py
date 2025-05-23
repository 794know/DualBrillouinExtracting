# D_noise_add.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/23
# This code is used for adding noise to the simulated clean dataset:
# Generation of dual-peaks BGS dataset with noise
# All the index can be modified in 'A_fiber_index.py'

import os
import numpy as np
import matplotlib.pyplot as plt

import A_fiber_index as Afb

if __name__ == '__main__':

    # Pumppower index
    pumppower_index = 0
    # SNR index
    snr_index = 3
    # SNR range
    snr_range = Afb.SNR_range
    print(f"当前信噪比索引: {snr_index}, 信噪比为: {snr_range[snr_index]} dB")

    # 生成数据集(Single Set)
    output_folder = f'dataset_SNR_{snr_range[snr_index]}dB'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载原始数据
    input_file = f'dataset_clean\Dataset_Pumppower_index_{pumppower_index}_SNR_Clean.npy'
    dataset = np.load(input_file)
    print(f"加载数据集: {input_file} \n Pumppower_index: {pumppower_index}")
    
    # 定义目标信噪比
    target_snr_db = snr_range[snr_index]
    target_snr = 10 ** (target_snr_db / 10)  # 将dB转换为线性比例

    # 获取数据的形状
    num_samples, sample_length = dataset.shape
    print(f"数据形状: {dataset.shape}")

    # 创建一个空数组用于存储添加噪声后的数据
    noisy_dataset = np.zeros_like(dataset)

    # 遍历每个样本，添加噪声
    for i in range(sample_length):
        signal = dataset[:, i]
        signal_power = np.max(signal)  # 计算信号功率
        noise_power = signal_power / target_snr  # 计算所需噪声功率
        noise = noise_power * np.random.randn(num_samples)  # 生成高斯噪声
        noisy_signal = signal + noise  # 添加噪声
        noisy_dataset[:, i] = noisy_signal

    # 保存添加噪声后的数据
    output_file = f'dataset_SNR_{snr_range[snr_index]}dB\Dataset_Pumppower_index_{pumppower_index}_SNR_{snr_range[snr_index]}dB.npy'
    np.save(output_file, noisy_dataset)

    print(f"添加噪声后的数据已保存到 {output_file}")
    
    
    # 绘制添加噪声后的数据的 Y-Z 图
    fixed_x = 62000  # 选择第 10 列（索引从 0 开始）
    noisy_dataset = np.load(output_file)
    dataset = np.load(r'dataset_clean\Dataset_Pumppower_index_0_SNR_Clean.npy')
    dataset_yz = noisy_dataset[:, fixed_x]
    label_yz = dataset[:, fixed_x]
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_yz, label=f'Dataset (X={fixed_x})')
    plt.plot(label_yz, 'ro-', label=f'Label (X={fixed_x})')
    plt.title('Dataset_Pumppower_index_0_SNR_3dB Y-Z Plane')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.legend()
    plt.grid(True)
    plt.show()
