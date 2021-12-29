import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
    
def trueParameter(observation_coords):
    dx = 12 / n_observations
    c = np.pi / 3
    n4 = n_observations / 4
    parameter = np.zeros((n_observations,))
    parameter[int(2*n4):int(3*n4)] = (dx + np.diff(np.sin((np.arange(0, 3 + 1e-15, dx)) * c)) / c) / dx ** 0.5
    parameter[int(n4):int(2*n4)] = parameter[int(3*n4):int(2*n4):-1]
    return parameter
    
def generateObservations(n_observations):
    if n_observations % 4 != 0:
        raise ValueError("n_observations must be a multiple of 4 for Phillips problem")
    dx = 12 / n_observations
    observation_coords = np.linspace(-6, 6, n_observations)
    observations = np.zeros((n_observations,))
    c = np.pi / 3
    for i in range(int(n_observations / 2), n_observations):
        t1 = -6 + i * dx
        t2 = t1 - dx
        observations[i] = t1 * (6 - np.abs(t1) / 2 )
        observations[i] += ((3 - np.abs(t1) / 2) * np.sin(c * t1) - 2 / c * (np.cos(c * t1) - 1)) / c 
        observations[i] += - t2 * (6 - np.abs(t2) / 2) 
        observations[i] += - ((3 - np.abs(t2) / 2) * np.sin(c * t2) - 2 / c * (np.cos(c * t2) - 1)) / c
        observations[n_observations - i] = observations[i]
    observations /= dx ** 0.5
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    dx = 12 / n_observations
    n4 = int(n_observations / 4)
    
    r1 = np.zeros((n_observations,))
    c = np.cos((np.arange(-1, n4 + 1) * 4 * np.pi / n_observations))
    r1[0:n4] = dx + 9 / (dx * np.pi ** 2) * (2 * c[1:n4+1] - c[0:n4] - c[2:n4+2])
    r1[n4] = dx / 2 + 9 / (dx * np.pi ** 2) * (np.cos(4 * np.pi / n_observations) - 1)
    forward_map = sp.linalg.toeplitz(r1)
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_observations = 1000
    noise_level = 0.01
    regularization = 200 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 1000
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
