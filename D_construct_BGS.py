# D_construct_BGS.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/19
# This code is used for the simulated dataset_clean of network:
# Generation of dual-peaks BGS dataset_clean
# All the index can be modified in 'A_fiber_index.py'

import os

import numpy as np
import A_fiber_index as Afb
import B_BGS_gene as Bgene
import B_calculation as Bcalc

if __name__ == '__main__':

    # Variable Table
    LineWidth_peak1 = np.linspace(50e-3, 60e-3, 3) # Peak1 lineWidth
    print('Peak1 LineWidth from:', f'{min(LineWidth_peak1)}', 'to', f'{max(LineWidth_peak1)}')
    print('Num:', f'{LineWidth_peak1.shape[0]}')
    LineWidth_peak2 = np.linspace(50e-3, 60e-3, 3) # Peak2 lineWidth
    Temperature_Change = np.linspace(1, 70, 70) # Temperature Change
    print('Temperature_Change from:', f'{min(Temperature_Change)}', 'to', f'{max(Temperature_Change)}')
    print('Num:', f'{Temperature_Change.shape[0]}')
    Strain_Change = np.linspace(20, 1400, 70) # Strain Change
    Ratio_of_2on1 = np.linspace(0.4, 0.6, 3) # Ratio of Two Peaks
    print('Peak Ratio from:', f'{min(Ratio_of_2on1)}', 'to', f'{max(Ratio_of_2on1)}')
    print('Num:', f'{Ratio_of_2on1.shape[0]}')

    # Input Power of Probe Light
    power_input_set = np.array([0.01, 0.70, 1.35, 1.95, 2.5, 2.98, 3.44, 3.90, 4.3, 4.68, 5], dtype=np.float64)

    # Array
    ## Original Sweeping Range
    Sweeping_Range = Afb.Sweep_v

    ## Total BGS number
    Total_Num = (LineWidth_peak1.shape[0] * LineWidth_peak2.shape[0] *
                 Temperature_Change.shape[0] * Strain_Change.shape[0] *
                 Ratio_of_2on1.shape[0])

    ## Empty Array
    Generated_Dual_BGS = np.zeros((Sweeping_Range.shape[0], Total_Num))
    Label_Of_Dataset = np.zeros((2, Total_Num))
    print('Dataset Shape:', f'{Generated_Dual_BGS.shape}')
    print('Label Shape:', f'{Label_Of_Dataset.shape}')
    # Load Pump Distribution
    file_dir = 'pump_data/'
    file_name_pump = 'distorted_pump_under_0.01-5_mW.npy'
    Distorted_pump = np.load(f'{file_dir}{file_name_pump}')
    print(Distorted_pump.shape)

    # Variable Statements
    ## Original Value Given
    Original_BFS1 = Afb.fiber_vB_1
    Original_BFS2 = Afb.fiber_vB_2

    Original_FWHM1 = Afb.fiber_FWHM_1
    Original_FWHM2 = Afb.fiber_FWHM_2

    Brillouin_gain = Afb.g0
    Ratio_of_Peaks = Afb.gain_ratio

    Changed_BFS1 = 0
    Changed_BFS2 = 0

    # 使用嵌套循环遍历所有自变量的组合
    Pump_index = 10 # 记录哪一个功率下的DBGS
    Pump_distribution = Distorted_pump[:, Pump_index]
    print(Pump_distribution.shape)
    index = 0  # 用于记录当前结果存储的位置

    for x1 in range(Ratio_of_2on1.shape[0]):
        for x2 in range(LineWidth_peak1.shape[0]):
            for x3 in range(LineWidth_peak2.shape[0]):
                for x4 in range(Temperature_Change.shape[0]):
                    for x5 in range(Strain_Change.shape[0]):

                        # Variables Giving
                        Ratio_of_Peaks = Ratio_of_2on1[x1]
                        Original_FWHM1 = LineWidth_peak1[x2]
                        Original_FWHM2 = LineWidth_peak2[x3]

                        # Changed_BFS_Calculation
                        Changed_BFS1, Changed_BFS2 = Bcalc.variables_to_bfs(Temperature_Change[x4], Strain_Change[x5])
                        Changed_BFS1 = Changed_BFS1 + Original_BFS1
                        Changed_BFS2 = Changed_BFS2 + Original_BFS2

                        # BGS Generation
                        Original_Dual_BGS_End = Bgene.dual_lorentzian(Sweeping_Range, Changed_BFS1, Original_FWHM1, Brillouin_gain,
                                                                      Changed_BFS2, Original_FWHM2, Brillouin_gain * Ratio_of_Peaks)
                        Distorted_Dual_BGS_End = Original_Dual_BGS_End * Pump_distribution

                        # Label
                        temperature_value = Temperature_Change[x4]
                        strain_value = Strain_Change[x5]

                        # [2×1]  NumPy array
                        Label_Temp = np.array(([temperature_value],[strain_value]))
                        Label_Temp = Label_Temp.flatten()

                        # Give
                        Generated_Dual_BGS[:, index] = Distorted_Dual_BGS_End
                        Label_Of_Dataset[:, index] = Label_Temp

                        # Index Plus 1
                        index += 1
                        print(f"循环{index+1}次")
    # Dataset Saving
    variable1 = "Dataset_Pumppower_index"
    variable2 = "Label_Pumppower_index"
    variable3 = "SNR"

    folder_path = "dataset_clean"  # 替换为你的目标文件夹路径
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 构建文件名，将变量信息嵌入其中
    file_name_dataset = f"{variable1}_{Pump_index}_{variable3}_Clean.npy"
    file_name_label = f"{variable2}_{Pump_index}_{variable3}_Clean.npy"

    # 构建完整的文件路径
    file_path_dataset = os.path.join(folder_path, file_name_dataset)
    file_path_label = os.path.join(folder_path, file_name_label)

    try:
        # 保存为 .npy 文件
        np.save(file_path_dataset, Generated_Dual_BGS)
        np.save(file_path_label, Label_Of_Dataset)
        print(f"文件已成功保存到：{file_path_dataset} 和 {file_path_label}")
    except Exception as e:
        print(f"保存文件时出错：{e}")