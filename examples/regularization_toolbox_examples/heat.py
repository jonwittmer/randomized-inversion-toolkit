import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
    
def trueParameter(observation_coords):
    dx = 1 / n_observations
    observation_coords = dx * (np.arange(1, n_observations + 1) - 0.5)
    parameter = np.zeros((n_observations,))
    for i in range(int(n_observations / 2)):
        ti = (i + 1) * 20 / n_observations # + 1 to convert to matlab indexing
        if ti < 2:
            parameter[i] = (0.75 * ti ** 2) / 4
        elif ti < 3:
            parameter[i] = 0.75 + (ti - 2) * (3 - ti)
        else:
            parameter[i] = 0.75 * np.exp(-(ti - 3) * 2)
    return parameter
    
def generateObservations(n_observations):
    dx = 1 / n_observations
    observation_coords = np.arange(dx / 2, 1, dx)
    true_parameter = trueParameter(n_observations)
    observations = buildForwardMatrix(n_observations) @ true_parameter
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    dx = 1 / n_observations
    observation_coords = dx * (np.arange(1, n_observations + 1) - 0.5)
    c = dx / (2 * np.pi ** 0.5)
    d = 1 / 4
    k = c * observation_coords ** (-1.5) * np.exp(-d / observation_coords)
    r = np.zeros((1, n_observations))
    r[0] = k[0]
    forward_map = sp.linalg.toeplitz(k, r)
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_observations = 500
    noise_level = 0.01
    regularization = 50 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 500
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'cg'
    
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    randomized_solver = Strategies.RMA(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    ls_solution = randomized_solver.solve()
    u1_solution = no_randomizaton_solver.solve()
    fig, ax = plt.subplots()
    ax.plot(observation_coords, true_parameter, label='true parameter')
    ax.plot(observation_coords, ls_solution, label=randomized_solver.name + ' solution')
    ax.plot(observation_coords, u1_solution, label='u_1 solution')
    ax.set_title('Heat: ' + str(n_random_samples) + ' samples')
    ax.legend()
    plt.show()
