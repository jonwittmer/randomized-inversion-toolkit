import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
    
def trueParameter(observation_coords):
    n_observations = observation_coords.shape[0]
    dx = 1 / n_observations 
    parameter = (dx ** (3/2)) * (np.arange(n_observations) + 0.5)
    return parameter
    
def generateObservations(n_observations):
    dx = 1 / n_observations
    observation_coords = (np.arange(n_observations) + 0.5) * dx
    observations = np.zeros((n_observations,))
    for i in range(n_observations):
        observations[i] = (dx ** (3 / 2)) * ((i + 1) - 0.5) * (((i + 1) ** 2 + i ** 2) * (dx ** 2) / 2 - 1) / 6
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    dx = 1 / n_observations
    t = 2 / 3
    forward_map = np.zeros((n_observations, n_observations))
    for i in range(n_observations):
        forward_map[i, i] = dx ** 2 * (((i + 1) ** 2 - (i + 1) + 0.25) * dx - ((i + 1) - t))
        for j in range(i-1):
            forward_map[i, j] = dx ** 2 * ((j + 1) - 0.5) * (((i + 1) - 0.5) * dx - 1)
    forward_map += np.tril(forward_map, -1).T
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_observations = 200
    noise_level = 0.01
    regularization = 6.5 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 50
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    randomized_solver = Strategies.RMA(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    ls_solution = randomized_solver.solve()
    u1_solution = no_randomizaton_solver.solve()
    fig, ax = plt.subplots()
    ax.plot(observation_coords, true_parameter, label='true parameter')
    ax.plot(observation_coords, ls_solution, label=randomized_solver.name + ' solution')
    ax.plot(observation_coords, u1_solution, label='u_1 solution')
    ax.set_title('Shaw: ' + str(n_random_samples) + ' samples')
    ax.legend()
    plt.show()
