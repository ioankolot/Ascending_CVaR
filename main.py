import math
from vqe import VQE
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from cvar import CVaR

#The example below is referred to the Number Partitioning problem.

#We will use linear ascending. This can be modified. An ascending function, as described in the paper works really well.
#You can change both values below.

ascending_factor = 0.03 #The ascending factor
initial_alpha = 0.01 #The starting value of alpha. 

#We prepare the random instance of the number partion problem.
number_of_qubits = 10
number_list = [np.random.randint(0, 100) for _ in range(number_of_qubits)]

layers = 2 #The number of layers of the hardware efficient ansatz VQE.
single_qubit_gates = 'ry'
two_qubit_gates = 'cz'
entanglement  = 'linear'
maxiter = int((layers+1)*3*number_of_qubits/2) #The maximum number of iterations for the optimiser of the Ascending-CVaR
steps = 16 #The number of times we ascend alpha.
init_thetas = [np.random.uniform(0, 2*np.pi) for _ in range((layers+1)*number_of_qubits)]


cvar_opt = CVaR(0.2, maxiter*steps, number_of_qubits, single_qubit_gates, two_qubit_gates, entanglement, layers, number_list)
print(cvar_opt.optimize(init_thetas))


cvar_ascending_opt = CVaR(initial_alpha, maxiter, number_of_qubits, single_qubit_gates, two_qubit_gates, entanglement, layers, number_list)
print(cvar_ascending_opt.optimize(init_thetas, 'ascending_cvar', options={'steps':15, 'ascending_factor':ascending_factor}))