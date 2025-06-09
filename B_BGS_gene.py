# B_BGS_gene
# Author: QYH
# Version: 1.2
# Date: 2025/06/09
# This code is the collection of BGS and pump distortion codes
# All the index can be modified in 'A_fiber_index.py'

import numpy as np
import matplotlib.pyplot as plt
import A_fiber_index as Afb
from scipy.integrate import quad

#- 定义单峰洛伦兹函数
def lorentzian(x1, x0, gamma, a):
    """
    x1: 输入数组
    x0: 峰的位置
    gamma: 峰的半高全宽
    A: 峰的幅度
    """
    return a * (gamma / 2)**2 / ((x1 - x0) ** 2 + (gamma / 2) ** 2)

#- 定义双峰洛伦兹函数
def dual_lorentzian(x1, x00, gamma1, a1, x11, gamma2, a2):
    """
    x1: 输入数组
    x00: 峰1的位置
    gamma1: 峰1的半高全宽
    A1: 峰1的幅度
    x11: 峰2的位置
    gamma2: 峰2的半高全宽
    A2: 峰2的幅度
    """
    dual_lorentzian_spectra=lorentzian(x1, x00, gamma1, a1)+lorentzian(x1, x11, gamma2, a2)
    return dual_lorentzian_spectra

#- 定义单个扫描频率下的频偏函数
def refractive_in_vb(sweeping_freq):
    """
    Calculate Delta Vb caused by sweeping process.

    Parameters:
    sweeping_freq (float or array-like): Sweeping frequency (GHz).

    Returns:
    float or array-like: Delta Vb (GHz).
    """
    # Constants
    refractive = Afb.refractive_index # Refractive index
    v_acoustic = Afb.velocity_acoustic # Acoustic velocity (m/s)
    v_light = Afb.velocity_light  # Speed of light (m/s)

    # Calculate Delta Vb
    delta_vb = 2 * sweeping_freq * (refractive**2) * v_acoustic / v_light
    return delta_vb

#- 定义积分函数
def fun_integral(probe_power, dx, dg):
    return -dg * probe_power * np.exp(-Afb.attenuation_Coefficient * (Afb.fiberLength - dx)) / Afb.effective_area

#- 定义泵浦消耗积分式函数
def pump_distortion(position_of_distortion, probe_power_input):
    sweep_v=Afb.Sweep_v
    depleted = np.zeros_like(sweep_v)
    for i in range(sweep_v.shape[0]):
        # gain_of_lower_peak1 = lorentzian(sweep_v, Afb.fiber_vB_1 - refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_1, Afb.g0)
        # gain_of_lower_peak2 = lorentzian(sweep_v, Afb.fiber_vB_2 - refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_2, Afb.g0 * Afb.gain_ratio)
        gain_of_pump_lower = dual_lorentzian(sweep_v, Afb.fiber_vB_1 - refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_1, Afb.g0,
                                             Afb.fiber_vB_2 - refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_2, Afb.g0 * Afb.gain_ratio)
        # gain_of_pump_lower = gain_of_lower_peak1 + gain_of_lower_peak2
        loss_of_upper_peak1 = lorentzian(sweep_v, Afb.fiber_vB_1 + refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_1, Afb.g0)
        loss_of_upper_peak2 = lorentzian(sweep_v, Afb.fiber_vB_2 + refractive_in_vb(sweep_v[i]), Afb.fiber_FWHM_2, Afb.g0 * Afb.gain_ratio)
        loss_of_pump_upper = loss_of_upper_peak1 + loss_of_upper_peak2
        net_gain_of_pump = gain_of_pump_lower - loss_of_pump_upper
        depleted[i] = -net_gain_of_pump[i]
    q = np.zeros_like(sweep_v)
    for j in range(q.shape[0]):
        # Don't mind the yellow '~~~~'
        q[j], _ = quad(lambda x: fun_integral(probe_power_input , x, depleted[j]), 0, position_of_distortion)
    return np.exp(q)

if __name__ == '__main__':

    # 参数设置 BGS Test
    x = Afb.Sweep_v  # x轴范围
    x0_1, x0_2 = Afb.fiber_vB_1, Afb.fiber_vB_2  # 两个峰的位置
    gamma_1, gamma_2 = Afb.fiber_FWHM_1, Afb.fiber_FWHM_2  # 两个峰的半高全宽
    A_1, A_2 = Afb.g0, Afb.gain_ratio * Afb.g0  # 两个峰的幅度
    y = lorentzian(x, x0_1, gamma_1, A_1) + lorentzian(x, x0_2, gamma_2, A_2)
    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="No pump distortion", color="blue")
    plt.xlabel("Sweeping Frequency (GHz)")
    plt.ylabel("Brillouin Gian")
    plt.title("Dual-peaks BGS")
    plt.legend()
    plt.grid(True)

    ################################################################################
    # 参数设置 BGS Test
    location_test = 20e3 # m
    probe_input = 0.01e-3 # mW
    pump_distribution = pump_distortion(location_test, probe_input)
    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(x, pump_distribution, label=f"Probe input = {probe_input} mW", color="blue")
    plt.xlabel("Sweeping Frequency (GHz)")
    plt.ylabel("Normalized pump power (a.u.)")
    plt.title("Pump distribution")
    plt.legend()
    plt.grid(True)
    ################################################################################
    # 显示所有图形
    plt.show()
