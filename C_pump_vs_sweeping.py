# D_dataset_construct
# Author: QYH
# Version: 1.0
# Date: 2025/04/28
# This code is used for the simulated dataset_clean of network:
# Generation of dual-peaks BGS dataset_clean
# All the index can be modified in 'A_fiber_index.py'

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import A_fiber_index as Afb
import B_BGS_gene as Bgene

# 设置全局字体样式
plt.rcParams['font.size'] = 12  # 设置默认字体大小
plt.rcParams['font.family'] = 'serif'  # 设置字体类型为 serif（如 Times New Roman）
plt.rcParams['font.serif'] = ['Arial']  # 指定具体的 serif 字体
plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗

# 色卡
colors_255 = np.array([
    [69, 42, 61],
    [0, 0, 0],#
    [68, 117, 122],   #
    [0, 0, 0],#
    [183, 181, 160],   #
    [0, 0, 0],#
    [238, 213, 183],   #
    [0, 0, 0],#
    [229, 133, 93],   #
    [0, 0, 0],#
    [212, 76, 60]  #
])

# 将 [0, 255] 范围的 RGB 值转换为 [0, 1] 范围
colors = colors_255 / 255.0

if __name__ == '__main__':

    power_input_set = np.array([0.01, 0.70, 1.35, 1.95, 2.5, 2.98, 3.44, 3.90, 4.3, 4.68, 5], dtype=np.float64)
    ### print(power_input_set.shape[0])
    sweeping_frequency = Afb.Sweep_v

    # Creating array for saving pump shape
    pump_distribution_under = np.zeros((sweeping_frequency.shape[0], power_input_set.shape[0]), dtype=np.float64)
    print(pump_distribution_under.shape[1])

    # Distorted pump distribution generation
    for i in range(pump_distribution_under.shape[1]):
        # Set the input_power of probe
        Probe_input = power_input_set[i] * 0.001 # mW
        Distorted_1 = Bgene.pump_distortion(20e3, Probe_input)
        pump_distribution_under[:, i] = Distorted_1

    folder_path = "pump_data"  # 替换为你的目标文件夹路径
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name_pumpdata = 'distorted_pump_under_0.01-5_mW'
    file_path_pumpdata = os.path.join(folder_path, file_name_pumpdata)

    np.save(f"{file_path_pumpdata}.npy", pump_distribution_under)
    print(f"数组已保存为{file_name_pumpdata}.npy")

    with open(f"{file_path_pumpdata}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pump_distribution_under)  # 写入二维数组的每一行
    print(f"数组已保存为{file_name_pumpdata}.csv")

    # 创建一个 600 个点的 x 轴数据
    x = np.linspace(10.651, 11.250, 600)
    # 保存数据(.npy and .csv)
    # 创建一个 3D 图形
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    y_values = np.array([0.01, 0.70, 1.35, 1.95, 2.5, 2.98, 3.44, 3.90, 4.3, 4.68, 5])
    # 遍历每一列，并绘制每条曲线的平面
    for i in range(0, pump_distribution_under.shape[1], 2):  # 遍历 11 列
        # 创建当前列的网格
        X = np.vstack([x, x])
        Y = np.full((2, len(x)), y_values[i])  # 使用物理值 y_values[i]
        Z = np.vstack([np.zeros_like(x), pump_distribution_under[:, i]])  # 从 Z=0 到当前列的值

        # 绘制当前列的平面
        ax.plot_surface(X, Y, Z, color=colors[i], edgecolor=colors[i], linewidth=1, alpha=1)
    # 自定义 X 轴和 Y 轴的范围
    ax.set_xlim(10.651, 11.250)  # 设置 X 轴范围
    ax.set_ylim(0, 5)  # 设置 Y 轴范围
    ax.set_zlim(0, 2.2)  # 设置 Z 轴范围

    # 设置标题和标签的字体样式
    title_font = {'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': 'darkblue', 'weight': 'normal', 'size': 12}

    # 设置刻度标签的字体样式
    ax.set_xticks(np.linspace(10.651, 11.250, 3))  # 设置 X 轴刻度位置
    ax.set_yticks(np.linspace(0, 5, 6))  # 设置 Y 轴刻度位置
    ax.set_zticks(np.linspace(0, 2, 5))  # 设置 Z 轴刻度位置

    ax.set_xticklabels(['10.65', '10.95', '11.25'], fontdict={'family': 'serif', 'size': 15})  # 自定义 X 轴刻度标签
    ax.set_yticklabels(['0', '1.35', '2.50', '3.44', '4.30', '5.0'], fontdict={'family': 'serif', 'size': 15})  # 自定义 Y 轴刻度标签
    ax.set_zticklabels(['0', '0.5', '1.0', '1.5', '2.0'], fontdict={'family': 'serif', 'size': 15})  # 自定义 Z 轴刻度标签

    # 调整刻度标签的位置
    ax.tick_params(axis='x', pad=10)  # 增加 X 轴刻度标签的间距
    ax.tick_params(axis='y', pad=10)  # 增加 Y 轴刻度标签的间距

    # 调整图形的布局
    ### plt.tight_layout()

    # 自定义网格颜色和数量
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0)
    ax.zaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1)
    ax.set_title('Pump Distribution vs. Different Input Power')

    # 设置视角
    ax.view_init(elev=21, azim=-57)
    file_name_pumppic = 'Pump_Distortion_vs_Input_Power'
    file_path_pumppic = os.path.join(folder_path, file_name_pumppic)
    # 保存图像并设置 DPI
    plt.savefig(f"{file_path_pumppic}.png", dpi=600)  # 设置 DPI 为 300
    print(f"图片已保存为{file_name_pumppic}.png")
    # 显示图形
    plt.show()