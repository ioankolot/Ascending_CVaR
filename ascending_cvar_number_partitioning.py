import math
from number_partioning_vqe import VQE
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

#The code below refers to the paper: An evolving objective function for improved variational quantum optimisation.

#The example below is referred to the Number Partitioning problem.

#We will use linear ascending. This can be modified. An ascending function, as described in the paper works really well.
#You can change both values below.
ascending_factor = 0.03 #The ascending factor
alpha = 0.01 #The starting value of alpha. 

#We prepare the random instance of the number partion problem.
number_of_qubits = 16
number_list = [np.random.randint(0, 500) for _ in range(number_of_qubits)]

layers = 1 #The number of layers of the hardware efficient ansatz VQE.
maxiter = int((layers+1)*3*number_of_qubits/2) #The maximum number of iterations for the optimiser for the Ascending-CVaR
steps = 16 #The number of times we ascend alpha.


#First of all we define the CVaR objective function as defined by Barkoutsos et al.
def CVaR(percentage, energies): #percentage = α , i.e the tail of the distribution
    len_list = len(energies)
    ak = math.ceil(len_list * percentage)
    cvar = 0
    for sample in range(ak):
        cvar += energies[sample]
    return cvar/ak

#We have to test how the Ascending-CVaR behaves in respect to other constant CVaR objective functions with different values of α.
#We work with α = 0.1, 0.2, 0.5, 1, where α = 1 corresponds to the expectation value, but you can play with different values of α. For our experiments you will need α > 0.1.

#We set four global variables to keep track of the constant CVaR optimisation in order to plot the results.
cvar_01_counter = 0
cvar_02_counter = 0
cvar_05_counter = 0
exp_value_counter = 0

constant_01_probabilities = [0]
constant_02_probabilities = [0]
constant_05_probabilities = [0]
exp_value_probabilities = [0]

def CVaR_optimisation(x, layers, alfa, type):
    if alfa == 0.1 and type =='const':
        global cvar_01_counter
        cvar_01_counter += 1
        if not cvar_01_counter % (3*number_of_qubits) and cvar_01_counter <= steps*maxiter:
            angles = list(x)
            vqe = VQE(number_of_qubits, number_list, angles, layers, 0.1)
            prob_of_optimal = vqe.probability_of_optimal()
            energies = vqe.exact_counts()
            cost = CVaR(alfa, energies)
            print(f'\nOn the {cvar_01_counter} iteration with a = {alfa} the probability of sampling the optimal solution is {prob_of_optimal} and the cost value is {cost}')
            constant_01_probabilities.append(prob_of_optimal)

    elif alfa == 0.2 and type =='const':
        global cvar_02_counter
        cvar_02_counter += 1
        if not cvar_02_counter % (3*number_of_qubits) and cvar_02_counter <= steps*maxiter:
            vqe = VQE(number_of_qubits, number_list, list(x), layers, 0.1)
            prob_of_optimal = vqe.probability_of_optimal()
            energies = vqe.exact_counts()
            cost = CVaR(alfa, energies)
            print(f'\nOn the {cvar_02_counter} iteration with a = {alfa} the probability of sampling the optimal solution is {prob_of_optimal} and the cost value is {cost}')
            constant_02_probabilities.append(prob_of_optimal)


    elif alfa == 0.5 and type =='const':
        global cvar_05_counter
        cvar_05_counter += 1
        if not cvar_05_counter % (3*number_of_qubits) and cvar_05_counter <= steps * maxiter:
            vqe = VQE(number_of_qubits, number_list, list(x), layers, 0.1)
            prob_of_optimal = vqe.probability_of_optimal()
            energies = vqe.exact_counts()
            cost = CVaR(alfa, energies)
            print(f'On the {cvar_05_counter} iteration with a = {alfa} the probability of sampling the optimal solution is {prob_of_optimal} and the cost value is {cost}')
            constant_05_probabilities.append(prob_of_optimal)

    elif alfa == 1 and type =='const':
        global exp_value_counter
        exp_value_counter += 1
        if not exp_value_counter % (3*number_of_qubits) and exp_value_counter <= steps*maxiter:
            vqe = VQE(number_of_qubits, number_list, list(x), layers, 0.1)
            prob_of_optimal = vqe.probability_of_optimal()
            energies = vqe.exact_counts()
            cost = CVaR(alfa, energies)
            print(f'On the {exp_value_counter} iteration with a = {alfa} the probability of sampling the optimal solution is {prob_of_optimal} and the cost value is {cost}')
            exp_value_probabilities.append(prob_of_optimal)

        
    angles = list(x)
    vqe = VQE(number_of_qubits, number_list, angles, layers, alfa)
    energies = vqe.exact_counts()
    cost = CVaR(alfa, energies)
    return cost



def ascending_optimization(layers):
    #We begin by initializing the same random parameters for all optimizers.
    thetas_init = [np.random.uniform(0, 2*np.pi) for _ in range((1+layers)*number_of_qubits)]
    thetas_init_01 = thetas_init.copy()
    thetas_init_02 = thetas_init.copy()
    thetas_init_05 = thetas_init.copy()
    thetas_init_expectation_value = thetas_init.copy()
    thetas_init_ascending = thetas_init.copy()


    #We set the global parameter alpha that will ascend slowly
    global alpha
    global cvar_01_counter
    global cvar_02_counter
    global cvar_05_counter
    global exp_value_counter


    ascending_probabilities = [0]
    ascending_optimal_thetas = [0 for _ in range((1+layers)*number_of_qubits)]

    #We minize all the different obejctive functions


    step = 0 #counter of steps of the ascending.
    while step < steps: #We slowly ascend alpha. You make it ascend up intill a=1 or stop the ascending sooner.
        if alpha > 1:
            alpha = 1


        #We begin with Ascending-CVaR

        ascending_optimization_object = scipy.optimize.minimize(CVaR_optimisation, x0 = tuple(thetas_init_ascending), args=(layers, alpha, 'non-const'), method='COBYLA', options={'maxiter':maxiter})
        ascending_objective_value = ascending_optimization_object.fun
        print(f' On the ascending optimization on {step} iteration with alpha = {alpha} the objective value is {ascending_objective_value}')
        ascending_optimal_thetas = ascending_optimization_object.x

        thetas_init_ascending = ascending_optimal_thetas #We continue the optimisation with initial angles the optimal angles of the previous iteration.
        ascending_vqe = VQE(number_of_qubits, number_list, ascending_optimal_thetas, layers, 0.1)
        #We put 0.1 above because we want the shots for comparing the probabilities to be equal for all different cases. Note that in the optimisation. the number
        #of shots is not equal but scales as 1/a

        ascending_probability = ascending_vqe.probability_of_optimal()
        ascending_probabilities.append(ascending_probability)# We keep track of the overlap with the optimal solution.
        print(f'and the probability of obtaining the optimal solution is {ascending_probability}')  


        step += 1
        alpha += ascending_factor   


    #Constant CVaR α = 0.1

    while cvar_01_counter <= steps*maxiter:

        constant_cvar_01_object = scipy.optimize.minimize(CVaR_optimisation, x0 = tuple(thetas_init_01), args=(layers, 0.1, 'const'), method='COBYLA',  options={'maxiter':steps*maxiter})
        thetas_init_01 = constant_cvar_01_object.x

    #Constant CVaR α = 0.2

    while cvar_02_counter <= steps*maxiter:
        constant_cvar_02_object = scipy.optimize.minimize(CVaR_optimisation, x0 = tuple(thetas_init_02), args=(layers, 0.2, 'const'), method='COBYLA',  options={'maxiter':steps*maxiter})
        thetas_init_02 = constant_cvar_02_object.x

    #Constant CVaR α = 0.5

    while cvar_05_counter <= steps*maxiter:
        constant_cvar_05_object = scipy.optimize.minimize(CVaR_optimisation, x0 = tuple(thetas_init_05), args=(layers, 0.5, 'const'), method='COBYLA',  options={'maxiter':steps*maxiter})
        thetas_init_05 = constant_cvar_05_object.x


    #Constant CVaR α = 1


    while exp_value_counter <= steps*maxiter:
        expectation_value_object = scipy.optimize.minimize(CVaR_optimisation, x0 = tuple(thetas_init_expectation_value), args=(layers, 1, 'const'), method='COBYLA',  options={'maxiter':steps*maxiter})
        thetas_init_expectation_value = expectation_value_object.x


    return ascending_probabilities


ascending_probabilities = ascending_optimization(layers)


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)  

x_axis = np.linspace(0, int(layers*steps*maxiter/number_of_qubits), steps+1)

plt.figure(figsize=(10, 7.5))    
ax = plt.subplot(111)


ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()
plt.ylim(0, 1)

plt.yticks(np.linspace(0, 1, 11), [str(x) + "%" for x in range(0, 101, 10)], fontsize=14)    
plt.xticks(fontsize=14)

for y in np.linspace(0.1, 1, 10):    
    plt.plot(x_axis, [y] * len(x_axis), "--", lw=0.5, color="black", alpha=0.3)  

plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")

alphas = ['Ascending-α', 'Expectation Value', 'α = 0.5', 'α = 0.2', 'α = 0.1']
probabilities_list = [ascending_probabilities, exp_value_probabilities, constant_05_probabilities, constant_02_probabilities, constant_01_probabilities]


for rank, column in enumerate(alphas):    
    plt.plot(x_axis,    
            probabilities_list[rank],    
            lw=2.5, color=tableau20[rank], label=alphas[rank])

plt.legend()
plt.ylabel('Probability of sampling an optimal solution.', fontsize = 14)
plt.xlabel('Normalised Optimiser Iterations', fontsize = 14)
plt.show()

