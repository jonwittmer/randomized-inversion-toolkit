import matplotlib.pyplot as plt
import numpy as np

def computeLCurve(true_parameter, no_randomization_builder):
    regs  = np.logspace(-8, 5, 20)
    lcurve_sols = []
    for reg_scale in regs:
        no_randomization_solver = no_randomization_builder(reg_scale)
        no_randomization_solver.solver.atol = 1e-5
        no_randomization_solver.solver.tol = 1e-5
        no_randomization_solver.solver.maxiter = 1000
        u1_solution = no_randomization_solver.solve().reshape(true_parameter.shape)
        lcurve_sols.append(np.linalg.norm(u1_solution - true_parameter) / np.linalg.norm(true_parameter))

    fig, ax = plt.subplots()
    ax.plot(regs, lcurve_sols)
    plt.show()
