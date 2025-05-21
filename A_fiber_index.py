# A_fiber_index of fiber
# Author: QYH
# Version: 1.0
# Date: 2025/04/21
# This code is the basic information of FUT
# 600 MHz

import numpy as np

#- Constants
refractive_index = 1.46  # Refractive index
velocity_acoustic = 5.759e3  # Acoustic velocity (m/s)
velocity_light = 3e8  # Speed of light (m/s)

#- Basic information
fiberLength = 20e3  # Length of Fiber; m
attenuation_Coefficient = 0.215 / (4.343 * 1000)  # Attenuation Coefficient
effective_area = 80e-12  # Effective Area; m^2

g0 = 2e-11  # Gain Coefficient of the First Peak; m/W
gain_ratio = 0.6 # Peak gain ratio of 2 peaks

fiber_vB_1 = 10.831  # BFS of the First Peak; GHz
fiber_vB_2 = 10.916  # BFS of the Second Peak; GHz
##- AKA room temperature BFSs
fiber_FWHM_1 = 54.8e-3  # FWHM of the First Peak; GHz
fiber_FWHM_2 = 53.5e-3  # FWHM of the Second Peak; GHz
##- AKA classic value

#- Temperature and strain
temperature_coefficient_Peak_1=1.9754e-3 # GHz/°C
strain_coefficient_Peak_1=0.0483e-3 # GHz/ue

temperature_coefficient_Peak_2=2.0351e-3 # GHz/°C
strain_coefficient_Peak_2=0.0493e-3 # GHz/ue

#- Sweep range
Sweep_v = np.linspace(10.651, 11.250, 600) # Simulation Sweep Range; GHz
# print(Sweep_v.shape[0])
