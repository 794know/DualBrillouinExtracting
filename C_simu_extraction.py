# C_simu_extraction
# Author: QYH
# Version: 1.0
# Date: 2025/04/24
# This code is the simulated extraction of CESM:
# LCF for two peaks
# All the index can be modified in 'A_fiber_index.py'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit


import A_fiber_index as Afb
import B_BGS_gene as Bgene
import B_calculation as Bcalc
from B_BGS_gene import dual_lorentzian

if __name__ == '__main__':
    Temp_set = np.linspace(1, 40, 40)
    ### print(Temp_set.shape[0])
    Strain_set = np.linspace(20, 800, 40)
    ### print(Strain_set.shape[0])
    Temperature_array = np.zeros((Temp_set.shape[0], Strain_set.shape[0]))
    Strain_array = np.zeros_like(Temperature_array)
    print(Temperature_array.shape)
    '''
    for i in range(Temp_set.shape[0]):
        for j in range(Strain_set.shape[0]):
            Combi_1 = Temp_set[i], Strain_set[j]
            ### print(Combi_1)
            deta_BFS_1 = Bcalc.variables_to_bfs(Combi_1[0], Combi_1[1])
            ### print(deta_BFS_1)

            # Distorted pump distribution generation
            Probe_input = 0.0002
            Distorted_1 = Bgene.pump_distortion(20e3, Probe_input)

            # Original BGS generation
            ori_BGS_1 = Bgene.dual_lorentzian(Afb.Sweep_v, Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                                              Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0)

            # Distorted dual-BGS generation
            distorted_BGS_1 = ori_BGS_1 * Distorted_1

            # Fitting of original
            Xaxis_of_this = Afb.Sweep_v
            Initial_fitt_vector = [Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                                   Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0]
            params_0, params_covariance_0 = curve_fit(dual_lorentzian, Xaxis_of_this, ori_BGS_1, Initial_fitt_vector)
            ### print(params_0)
            ### print(params_covariance_0)
            Fitted_dual_BGS_0 = dual_lorentzian(Xaxis_of_this, params_0[0], params_0[1], params_0[2],
                                              params_0[3], params_0[4], params_0[5])

            # Fitting of distorted
            Initial_fitt_vector = [Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                                   Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0]
            params_1, params_covariance_1 = curve_fit(dual_lorentzian, Xaxis_of_this, distorted_BGS_1, Initial_fitt_vector)
            ### print(params_1)
            ### print(params_covariance_1)
            Fitted_dual_BGS = dual_lorentzian(Xaxis_of_this, params_1[0], params_1[1], params_1[2],
                                              params_1[3], params_1[4], params_1[5])
            # Calculation
            RMSE_of_Peaks_0 = abs(params_0[0] - Afb.fiber_vB_1), abs(params_0[3] - Afb.fiber_vB_2)
            print("仿真ori_BGS：Peak1 = ", f"{format(Afb.fiber_vB_1 + deta_BFS_1[0], '.5f')}")
            print("仿真ori_BGS：Peak2 = ", f"{format(Afb.fiber_vB_2 + deta_BFS_1[1], '.5f')}")
            print("仿真ori_fitted_BGS：Peak1 = ", f"{format(params_0[0], '.5f')}")
            print("仿真ori_fitted_BGS：Peak2 = ", f"{format(params_0[3], '.5f')}")
            RMSE_of_TS_0 = Bcalc.deta_bfs_to_variables(RMSE_of_Peaks_0[0], RMSE_of_Peaks_0[1])
            print("仿真ori_BGS温度=", f"{format(Combi_1[0], '.2f')}", "℃")
            print("仿真ori_BGS应力=", f"{format(Combi_1[1], '.2f')}", "μe")
            print("仿真ori_fitted_BGS温度=", f"{format(RMSE_of_TS_0[0], '.2f')}", "℃")
            print("仿真ori_fitted_BGS应力=", f"{format(RMSE_of_TS_0[1], '.2f')}", "μe")

            RMSE_of_Peaks_1 = abs(params_1[0] - Afb.fiber_vB_1), abs(params_1[3] - Afb.fiber_vB_2)
            print("仿真distorted_fitted_BGS：Peak1 = ", f"{format(params_1[0], '.5f')}")
            print("仿真distorted_fitted_BGS：Peak2 = ", f"{format(params_1[3], '.5f')}")
            RMSE_of_TS_1 = Bcalc.deta_bfs_to_variables(RMSE_of_Peaks_1[0], RMSE_of_Peaks_1[1])
            print("仿真distorted_fitted_BGS温度=", f"{format(RMSE_of_TS_1[0], '.2f')}", "℃")
            print("仿真distorted_fitted_BGS应力=", f"{format(RMSE_of_TS_1[1], '.2f')}", "μe")
            Temperature_array[i][j] = RMSE_of_TS_1[0]
            Strain_array[i][j] = RMSE_of_TS_1[1]
    # Error calculation
    Temperature_error_array = np.zeros_like(Temperature_array)
    Strain_error_array = np.zeros_like(Strain_array)
    for i in range(Temperature_array.shape[0]):
        for j in range(Temperature_array.shape[1]):
            Temperature_error_array[i][j] = abs(Temperature_array[i][j] - Temp_set[i])
            Strain_error_array[i][j] = abs(Strain_array[i][j] - Strain_set[j])
    # 创建X和Y的坐标网格
    x = np.linspace(0, 40, 40)  # X轴范围从0到40，共40个点
    y = np.linspace(0, 800, 40)  # Y轴范围从0到800，共40个点
    X, Y = np.meshgrid(x, y)  # 生成网格

    # 使用pcolormesh绘制X-Y平面图
    plt.figure(figsize=(8, 6))
    #
    plt.pcolormesh(X, Y, Temperature_error_array, shading='auto', cmap='viridis', vmin=0, vmax=50)  # 使用pcolormesh绘制
    plt.colorbar(label='Value')  # 添加颜色条
    plt.title("Temperature Error")
    plt.xlabel("Temperature (℃)")
    plt.ylabel("Strain (μe)")
    plt.xlim(0, 40)  # 设置X轴范围
    plt.ylim(0, 800)  # 设置Y轴范围
    #
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Strain_error_array, shading='auto', cmap='viridis', vmin=0, vmax=2000)  # 使用pcolormesh绘制
    plt.colorbar(label='Value')  # 添加颜色条
    plt.title("Stain Error")
    plt.xlabel("Temperature (℃)")
    plt.ylabel("Strain (μe)")
    plt.xlim(0, 40)  # 设置X轴范围
    plt.ylim(0, 800)  # 设置Y轴范围
    plt.show()
    '''

    Combi_1 = Temp_set[23], Strain_set[10]
    ### print(Combi_1)
    deta_BFS_1 = Bcalc.variables_to_bfs(Combi_1[0], Combi_1[1])
    ### print(deta_BFS_1)

    # Distorted pump distribution generation
    Probe_input = 0.00298
    Distorted_1 = Bgene.pump_distortion(20e3, Probe_input)

    # Original BGS generation
    ori_BGS_1 = Bgene.dual_lorentzian(Afb.Sweep_v, Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                                      Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0)

    # Distorted dual-BGS generation
    distorted_BGS_1 = ori_BGS_1 * Distorted_1

    # Fitting of original
    Xaxis_of_this = Afb.Sweep_v
    Initial_fitt_vector = [Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                           Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0]
    params_0, params_covariance_0 = curve_fit(dual_lorentzian, Xaxis_of_this, ori_BGS_1, Initial_fitt_vector)
    ### print(params_0)
    ### print(params_covariance_0)
    Fitted_dual_BGS_0 = dual_lorentzian(Xaxis_of_this, params_0[0], params_0[1], params_0[2],
                                        params_0[3], params_0[4], params_0[5])

    # Fitting of distorted
    Initial_fitt_vector = [Afb.fiber_vB_1 + deta_BFS_1[0], Afb.fiber_FWHM_1, Afb.g0,
                           Afb.fiber_vB_2 + deta_BFS_1[1], Afb.fiber_FWHM_2, Afb.gain_ratio * Afb.g0]
    params_1, params_covariance_1 = curve_fit(dual_lorentzian, Xaxis_of_this, distorted_BGS_1, Initial_fitt_vector)
    ### print(params_1)
    ### print(params_covariance_1)
    Fitted_dual_BGS = dual_lorentzian(Xaxis_of_this, params_1[0], params_1[1], params_1[2],
                                      params_1[3], params_1[4], params_1[5])
    ## Plotting
    fig = plt.figure(figsize=(12,3.5))
    ### Sub1
    ax1 = plt.subplot(131)
    plt.plot(Afb.Sweep_v, Distorted_1, label=f"Power = {Probe_input * 1000} mW", color="red")
    plt.xlabel("Sweeping Frequency (GHz)")
    plt.ylabel("Normalized pump power (a.u.)")
    plt.title("Distorted Pump")
    plt.legend()
    plt.grid(True)
    ### Sub2
    ax2 = plt.subplot(132)
    plt.plot(Afb.Sweep_v, ori_BGS_1,
             label=f"T = {format(Combi_1[0], '.1f')} ℃" + "\n" + f"S = {format(Combi_1[1], '.1f')} μe", color="blue")
    plt.plot(Afb.Sweep_v, Fitted_dual_BGS_0,
             label="Fitted", color="lightblue", linestyle="--")
    plt.xlabel("Sweeping Frequency (GHz)")
    plt.ylabel("Brillouin Gian (a.u.)")
    plt.title("Original BGS")
    plt.legend()
    plt.grid(True)
    ### Sub3
    ax3 = plt.subplot(133)
    plt.plot(Afb.Sweep_v, distorted_BGS_1, label=f"Distorted", color="orange")
    plt.plot(Afb.Sweep_v, Fitted_dual_BGS, label="Fitted", color="yellow", linestyle="--")
    plt.xlabel("Sweeping Frequency (GHz)")
    plt.ylabel("Brillouin Gain (a.u.)")
    plt.title("Distorted Dual-BGS")
    plt.legend()
    plt.grid(True)
    ## Whole Fig
    fig.tight_layout(h_pad=1)
    plt.show()

    # Calculation
    RMSE_of_Peaks_0 = abs(params_0[0] - Afb.fiber_vB_1), abs(params_0[3] - Afb.fiber_vB_2)
    print("仿真ori_BGS：Peak1 = ", f"{format(Afb.fiber_vB_1 + deta_BFS_1[0], '.5f')}")
    print("仿真ori_BGS：Peak2 = ", f"{format(Afb.fiber_vB_2 + deta_BFS_1[1], '.5f')}")
    print("仿真ori_fitted_BGS：Peak1 = ", f"{format(params_0[0], '.5f')}")
    print("仿真ori_fitted_BGS：Peak2 = ", f"{format(params_0[3], '.5f')}")
    RMSE_of_TS_0 = Bcalc.deta_bfs_to_variables(RMSE_of_Peaks_0[0], RMSE_of_Peaks_0[1])
    print("仿真ori_BGS温度=", f"{format(Combi_1[0], '.2f')}", "℃")
    print("仿真ori_BGS应力=", f"{format(Combi_1[1], '.2f')}", "μe")
    print("仿真ori_fitted_BGS温度=", f"{format(RMSE_of_TS_0[0], '.2f')}", "℃")
    print("仿真ori_fitted_BGS应力=", f"{format(RMSE_of_TS_0[1], '.2f')}", "μe")

    RMSE_of_Peaks_1 = abs(params_1[0] - Afb.fiber_vB_1), abs(params_1[3] - Afb.fiber_vB_2)
    print("仿真distorted_fitted_BGS：Peak1 = ", f"{format(params_1[0], '.5f')}")
    print("仿真distorted_fitted_BGS：Peak2 = ", f"{format(params_1[3], '.5f')}")
    RMSE_of_TS_1 = Bcalc.deta_bfs_to_variables(RMSE_of_Peaks_1[0], RMSE_of_Peaks_1[1])
    print("仿真distorted_fitted_BGS温度=", f"{format(RMSE_of_TS_1[0], '.2f')}", "℃")
    print("仿真distorted_fitted_BGS应力=", f"{format(RMSE_of_TS_1[1], '.2f')}", "μe")

