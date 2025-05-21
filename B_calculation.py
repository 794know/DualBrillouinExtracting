# B_calculation
# Author: QYH
# Version: 1.0
# Date: 2025/04/21
# This code is the collection of the calculation functions including:
# Vari ---> BFS_changes
# BFS_changes ---> Vari
# LCF for two peaks
# All the index can be modified in 'A_fiber_index.py'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import A_fiber_index as Afb
import B_BGS_gene as Bgene

#- Variables -> BFS
def variables_to_bfs(tempchange, strainchange):
    detaBFS1 = Afb.temperature_coefficient_Peak_1 * tempchange + Afb.strain_coefficient_Peak_1 * strainchange
    detaBFS2 = Afb.temperature_coefficient_Peak_2 * tempchange + Afb.strain_coefficient_Peak_2 * strainchange
    return detaBFS1, detaBFS2

#- BFS -> Variables
def deta_bfs_to_variables(detaBFS1, detaBFS2):
    # Coefficients matrix
    # Input should be GHz
    A = np.array([
        [Afb.temperature_coefficient_Peak_1, Afb.strain_coefficient_Peak_1],
        [Afb.temperature_coefficient_Peak_2, Afb.strain_coefficient_Peak_2]
    ])

    # BFS changes vector
    b = np.array([detaBFS1, detaBFS2])

    # Solve the linear system
    solution = np.linalg.solve(A, b)

    # Extract temperature and strain changes
    tempchange, strainchange = solution

    return tempchange, strainchange

if __name__ == '__main__':
    # Test1
    Combine_1 = 10, 300
    print(Combine_1)
    Results_1 = variables_to_bfs(Combine_1[0], Combine_1[1])
    print(Results_1)
    # Test2
    Results_2 = deta_bfs_to_variables(Results_1[0], Results_1[1])
    print(Results_2)
    # Test3
    Test_curve = Bgene.dual_lorentzian(np.linspace(1,400,400), 55,60,1,102,45,0.59)
    print(Test_curve.shape[0])
    mid1 = np.linspace(1,Test_curve.shape[0], Test_curve.shape[0])
    print(mid1.shape[0])
    p0 = [59, 42, 1, 87, 55, 0.5]
    params, params_covariance = curve_fit(Bgene.dual_lorentzian, mid1, Test_curve, p0)
    print(params)
