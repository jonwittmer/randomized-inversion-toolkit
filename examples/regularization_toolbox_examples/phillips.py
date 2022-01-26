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
    random_vector_generator = np.random.multivariate_normal
    
    observation_coords, observations = generateObservations(n_observations)
    true_parameter = trueParameter(observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    problem_name = 'Phillips'

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
                        f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-0.06, 0.08]})

    writeLatexTables(results, f'{problem_name}_table.tex')
