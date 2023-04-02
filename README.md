# Ascending-CVaR

Qiskit implementation of the paper "Evolving objective function for improved variational quantum optimization"  (Phys. Rev. Research 4, 023225). Authors: Ioannis Kolotouros, Petros Wallden.


## Description
Ascending CVaR refers to an (evolving) objective function that can be used for combinatorial optimization problems when tackled with a variational quantum algorithm. The idea proposed by Barkoutsos et al. [1] was to construct a parameterized quantum state, measure it, and then keep only a percentage of outcomes that correspond to the lowest energies. Then at each iteration, make this percentage (given by the number α) as small as possible. In our proposed method we allow α to slowly increase, until the optimizer sees the whole distribution, achieving quantum states with a high overlap with the optimal solution. Details and motivation of our method can be found in [2].

[1] P. K. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner, Improving variational quantum optimization using CVaR, Quantum 4, 256 (2020).
[2] Ioannis Kolotouros and Petros Wallden. “Evolving objective function for improved variational quantum optimization”. Phys. Rev. Res. 4, 023225 (2022).


