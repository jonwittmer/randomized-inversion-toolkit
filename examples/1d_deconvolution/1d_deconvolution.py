import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies

# returns a unit basis vector with 1 in the index position
def canonicalBasisVector(size, index):
    vec = np.zeros((size,))
    vec[index] = 1
    return vec
    
def trueParameter(alpha, observation_coords):
    return np.sin(alpha * observation_coords) + np.cos(alpha * observation_coords)

def getPsf(dx):
    # controls the width of psf
    a = 0.1

    psf_half_width = np.ceil(a / dx)
    psf_coords = np.arange(-psf_half_width, psf_half_width + 1) * dx
    psf = (psf_coords - a)**2 * (psf_coords + a)**2

    # normalize psf to energy is not lost
    psf = psf / np.sum(psf)
    print('sum of psf: {}'.format(np.sum(psf)))
    return psf
    
def generateObservations(n_observations, alpha):
    dx = 1 / n_observations
    psf = getPsf(dx)
    observation_coords = np.linspace(0, 1, n_observations)
    observations = sp.ndimage.convolve1d(trueParameter(alpha, observation_coords), psf, mode='reflect')
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    psf = getPsf(1 / n_observations)
    forward_map = np.zeros((n_observations, n_observations))
    for i in range(n_observations):
        unit_action = sp.ndimage.convolve1d(canonicalBasisVector(n_observations, i), psf, mode='reflect')
        forward_map[:, i] = np.reshape(unit_action, (n_observations,))
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)

    # problem setup
    n_observations = 500
    alpha = 2 * np.pi
    noise_level = 0.05
    regularization = 5 # we will use identiy prior_covariance, parameterized by scalar given here
    n_random_samples = 500
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations, alpha)
    true_parameter = trueParameter(alpha, observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    
    randomized_solver = Strategies.ENKF(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, n_random_samples, solver_type)
    ls_solution = randomized_solver.solve()
    fig, ax = plt.subplots()
    ax.plot(observation_coords, true_parameter, label='true parameter')
    ax.plot(observation_coords, ls_solution, label=randomized_solver.name + ' solution')
    ax.set_title(str(n_random_samples) + ' samples')
    ax.legend()
    plt.show()
