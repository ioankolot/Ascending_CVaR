import math
from vqe import VQE
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


class CVaR():

    def __init__(self, alpha, maxiter, number_of_qubits, single_qubit_gates, two_qubit_gates, entanglement, layers, number_list):

        self.alpha = alpha
        self.maxiter = maxiter
        self.number_of_qubits = number_of_qubits
        self.single_qubit_gates = single_qubit_gates
        self.two_qubit_gates = two_qubit_gates
        self.entanglement = entanglement
        self.number_list = number_list
        self.layers = layers


    def get_CVaR(self, angles, alpha):

        
        vqe = VQE(self.number_of_qubits, angles, self.single_qubit_gates, self.two_qubit_gates, self.entanglement, self.layers, self.number_list, alpha)
        energies = vqe.exact_counts()


        num_samples = math.ceil(len(energies)*alpha)
        cvar = np.sum([energies[sample] for sample in range(num_samples)])/num_samples


        return cvar


    def optimize(self, initial_angles, type = 'constant', options= None):
        

        if type == 'constant':

            optimization_object = scipy.optimize.minimize(self.get_CVaR, x0=tuple(initial_angles), args=(self.alpha),
                                                                        method = 'COBYLA', options={'maxiter':self.maxiter})
            
        elif type == 'ascending_cvar':
            
            step = 1
            alpha = self.alpha
            while step < options['steps']:


                optimization_object = scipy.optimize.minimize(self.get_CVaR, x0=tuple(initial_angles), args=(alpha),
                                                                        method='COBYLA', options={'maxiter':self.maxiter})
      
                initial_angles = optimization_object.x
                alpha += options['ascending_factor']
                step += 1

        optimal_angles = optimization_object.x
        vqe = VQE(self.number_of_qubits, optimal_angles, self.single_qubit_gates, self.two_qubit_gates, self.entanglement, self.layers, self.number_list, 0.1)
        prob_of_sampling_optimal_solution = vqe.probability_of_optimal()
    
        return prob_of_sampling_optimal_solution

