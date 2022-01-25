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
    n_observations = 1000
    noise_level = 0.01
    regularization = 200 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'cg'
    problem_name = 'Gravity'

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
                        f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-1, 1.5]})

    writeLatexTables(results, f'{problem_name}_table.tex')
