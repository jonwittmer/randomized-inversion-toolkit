import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
    
def trueParameter(observation_coords):
    a1 = 2
    c1 = 6
    t1 = 0.8
    a2 = 1
    c2 = 2
    t2 = -0.5
    parameter = a1 * np.exp(-c1 * (observation_coords - t1) ** 2) + a2 * np.exp(-c2 * (observation_coords - t2) ** 2);
    return parameter
    
def generateObservations(n_observations):
    if n_observations % 2 != 0:
        raise ValueError("n_observations must be a multiple of 2")
    dx = np.pi / n_observations
    observation_coords = np.arange(-np.pi / 2, np.pi / 2, dx)
    observation_coords = -np.pi / 2 + dx * np.arange(0.5, n_observations + 0.5, 1)
    observations = buildForwardMatrix(n_observations) @ trueParameter(observation_coords)
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    dx = np.pi / n_observations
    observation_coords = np.arange(-np.pi / 2, np.pi / 2, dx)
    observation_coords = -np.pi / 2 + dx * np.arange(0.5, n_observations + 0.5, 1)
    forward_map = np.zeros((n_observations, n_observations))
    cosx = np.cos(observation_coords)
    pi_sinx = np.pi * np.sin(observation_coords)
    for i in range(n_observations // 2):
        for j in range(i, n_observations - i):
            ss = pi_sinx[i] + pi_sinx[j]
            forward_map[i,j] = ((cosx[i] + cosx[j]) * np.sin(ss) / ss) ** 2;
            forward_map[n_observations - j - 1 , n_observations - i - 1] = forward_map[i,j]
        forward_map[i, n_observations - i - 1 ] = (2 * cosx[i]) ** 2;
    forward_map += np.triu(forward_map, 1).T
    forward_map *= dx
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_observations = 200
    noise_level = 0.01
    regularization = 200 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 2500
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
