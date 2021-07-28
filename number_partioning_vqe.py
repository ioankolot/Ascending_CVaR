from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.visualization import *
import numpy as np



class VQE():
    def __init__(self, number_of_qubits, number_list, thetas, layers, alpha):

        self.number_of_qubits = number_of_qubits
        self.number_list = number_list
        self.init_thetas = thetas[0:self.number_of_qubits]
        self.thetas = thetas
        self.layers = layers
        self.alpha = alpha
        self.shots = int(1000/alpha)

        self.qreg = QuantumRegister(self.number_of_qubits, name = 'q')
        self.creg = ClassicalRegister(self.number_of_qubits, name = 'c')
        self.vqe = QuantumCircuit(self.qreg, self.creg)

        for qubit in range(self.number_of_qubits):
            self.vqe.ry(self.init_thetas[qubit], qubit)

        self.vqe.barrier()

        for layer in range(layers):
            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1 < qubit2:
                        self.vqe.cz(qubit1, qubit2)

            for qubit in range(self.number_of_qubits):
                self.vqe.ry(self.thetas[(layer+1)*self.number_of_qubits + qubit], qubit)

        self.vqe.barrier()

        self.vqe.measure(range(self.number_of_qubits), self.creg)
        self.counts = execute(self.vqe, Aer.get_backend('qasm_simulator'), shots = self.shots).result().get_counts()

    def get_expected_value(self):
        avr_c = 0
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y)
            avr_c += self.counts[sample] * tmp_eng
        energy_expectation = avr_c/self.shots
        return energy_expectation

    def cost_hamiltonian(self, x):
        spins = []
        for i in x[::-1]:
            spins.append(int(i))
        total_energy = 0
        for i in range(self.number_of_qubits):
            total_energy += self.number_list[i] * self.sigma(spins[i])
        total_energy = total_energy**2
        return total_energy

    def sigma(self, z):
        if z == 0:
            value = 1
        elif z == 1:
            value = -1
        return value

    def best_cost_brute(self):
        best_cost = np.inf
        for b in range(2**self.number_of_qubits):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(self.number_of_qubits)))]
            cost = 0
            for i in range(self.number_of_qubits):
                cost += self.number_list[i] * (2*x[i] - 1)
            cost = cost**2
            if best_cost > cost:
                best_cost = cost
        return best_cost

    def probability_of_optimal(self):
        optimal_solution = self.best_cost_brute()
        energies = self.exact_counts()
        print(f'The optimal objective value is {optimal_solution}')
        total_counts_of_optimal = 0
        for energy in energies:
            if energy == optimal_solution:
                total_counts_of_optimal += 1
        return total_counts_of_optimal/self.shots


    def exact_counts(self):
        energies = []
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y)
            for num in range(self.counts[sample]):
                energies.append(tmp_eng)
        energies.sort(reverse=False)
        return energies
