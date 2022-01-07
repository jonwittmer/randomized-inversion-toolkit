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
    return np.sin(np.pi * observation_coords) + 0.5 * np.sin(2 * np.pi * observation_coords)
    
def generateObservations(n_observations):
    dx = 1 / n_observations
    observation_coords = dx * (np.arange(1, n_observations + 1) - 0.5)
    true_parameter = trueParameter(n_observations)
    observations = buildForwardMatrix(n_observations) @ true_parameter
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    dx = 1 / n_observations
    ds = 1 / n_observations
    observation_coords = dx * (np.arange(1, n_observations + 1) - 0.5)
    s_observation_coords = ds * (np.arange(1, n_observations + 1) - 0.5)
    X, S = np.meshgrid(observation_coords, s_observation_coords)
    forward_map = dx * 0.25 * np.ones((n_observations, n_observations))
    forward_map = forward_map / ((0.25 ** 2 + (S - X) ** 2) ** 1.5)
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)

    # problem setup
    n_observations = 500
    noise_level = 0.05
    regularization = 200 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 100
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    
    randomized_solver = Strategies.RMA(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    ls_solution = randomized_solver.solve()
    fig, ax = plt.subplots()
    ax.plot(observation_coords, true_parameter, label='true parameter')
    ax.plot(observation_coords, ls_solution, label=randomized_solver.name + ' solution')
    ax.set_title('Gravity: ' + str(n_random_samples) + ' samples')
    ax.legend()
    plt.show()
