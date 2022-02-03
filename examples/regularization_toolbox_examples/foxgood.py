import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
from src.random_sampling import scaledIdentityCovGenerator
from utils.generate_figures import generateFigures
from utils.generate_tables import writeLatexTables
    
def trueParameter(observation_coords):
    return np.linspace(0, 1, n_observations)
    
def generateObservations(n_observations):
    dx = 1 / n_observations
    observation_coords = np.linspace(0, 1, n_observations)
    observations = ((1 + observation_coords ** 2) ** 1.5 - observation_coords ** 3) / 3
    return observation_coords, observations

def buildForwardMatrix(n_observations):
    column_observation_coords = np.linspace(0, 1, n_observations).reshape((n_observations, 1))
    forward_map = column_observation_coords **2 @ np.ones((1, n_observations))
    forward_map += np.ones((n_observations, 1)) @ (column_observation_coords.T ** 2)
    forward_map = 1 / n_observations * forward_map ** (0.5)
    return forward_map


if __name__ == '__main__':
    np.random.seed(20)

    # problem setup
    n_observations = 1000
    noise_level = 0.01
    regularization = 5 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = scaledIdentityCovGenerator
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    noise_std = noise_level * np.amax(observations)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    problem_name = 'Foxgood'

    # generate u1 solution only once
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_std**2), 0, regularization, random_vector_generator, 0, solver_type)
    u1_solution = no_randomizaton_solver.solve()
    
    n_random_samples = [10, 100, 1000, 10000]
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
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_std**2), 0, regularization, random_vector_generator, samples, solver_type)
            rand_solutions.append(randomized_solver.solve())
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
                        f"figures/{problem_name}/{randomized_solver.name}.pdf", lims={"ylim": [-1, 1.25]})

    writeLatexTables(results, f'{problem_name}_table.tex')
