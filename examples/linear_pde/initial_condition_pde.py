import os
import sys
sys.path.append('../../')
import json

import numpy as np
import scipy as sp
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle

from src.randomization_strategies import Strategies
from utils.generate_figures import generateFigures
from utils.generate_tables import writeLatexTables
    
def trueParameter():
    return scipy.io.loadmat('initial_condition.mat')['x']
    
def loadObservations():
    return scipy.io.loadmat('observation.mat')['x']

def loadForwardMatrix():
    return scipy.io.loadmat('linear_operator.mat')['A']

def loadBilaplacianPrior():
    return scipy.io.loadmat('bilaplacian_prior.mat')['x']


if __name__ == '__main__':
    np.random.seed(20)

    # for now, we are using data and operators that were exported for use in matlab
    # so load the files first and infer dimensions
    true_parameter = trueParameter().reshape((-1))
    observations = loadObservations().reshape((-1))
    forward_map = loadForwardMatrix()
    regularization = loadBilaplacianPrior()
    print(regularization.shape)
    
    # problem setup
    n_observations = observations.shape[0]
    noise_level = 0.01
    reg_param = 1
    random_vector_generator = np.random.multivariate_normal
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    noise_std = noise_level * np.amax(np.abs(observations))

    # pre-compute factorization of covariance matrices so that we don't need to
    # do an SVD every time we want to draw a sample (numpy implementation does this)
    #def sampleFromGaussianPrefactored(mean_vec, cov_half, n_samples)

    solver_type = 'direct'
    problem_name = 'initial_condition_pde'

    # generate u1 solution only once
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_std**2), 0, reg_param * regularization, random_vector_generator, 0, solver_type)
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
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_std**2), 0, reg_param * regularization, random_vector_generator, samples, solver_type)
            randomized_solver.solver.maxiter = 500
            rand_solutions.append(randomized_solver.solve())
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": [], "solutions": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            results[randomized_solver.name]["solutions"].append(rand_solutions[-1])
            print(f"{randomized_solver.name}:  N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        #generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
        #                f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-1.75, 1.75]})
    with open('results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    writeLatexTables(results, f'{problem_name}_table.tex')
    
