import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
from utils.generate_figures import generateFigures
from utils.generate_tables import writeLatexTables
    
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
    n_observations = 1000
    noise_level = 0.01
    regularization = 6.5 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    problem_name = 'Deriv2'

    # generate u1 solution only once
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, 0, solver_type)
    u1_solution = no_randomizaton_solver.solve()
    
    n_random_samples = [10, 100, 500, 1000]
    test_strategies = [
        Strategies.RMAP,
        Strategies.RMA,
        Strategies.RMA_RMAP,
        Strategies.RS_U1,
         Strategies.RS,
        Strategies.ENKF,
    ]
    results = {}
    
    for curr_strategy in test_strategies:
        rand_solutions = []
        rand_labels = []
        for samples in n_random_samples:
            rand_labels.append(f"N = {samples}")
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, samples, solver_type)
            rand_solutions.append(randomized_solver.solve())
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
                        f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-.01, .01]})

    writeLatexTables(results, f'{problem_name}_table.tex')
