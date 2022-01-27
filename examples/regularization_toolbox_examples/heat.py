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
    n_observations = 1000
    noise_level = 0.01
    regularization = 50 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = np.random.multivariate_normal
    solver_type = 'direct'
    problem_name = 'Heat'
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)

    # generate u1 solution only once
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, 0, solver_type)
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
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, samples, solver_type)
            rand_solutions.append(randomized_solver.solve())
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
                        f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-1, 1]})
        
    writeLatexTables(results, f'{problem_name}_table.tex')

