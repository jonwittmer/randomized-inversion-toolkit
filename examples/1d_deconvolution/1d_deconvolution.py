import os
import sys
sys.path.append('../../')
import json

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

from src.randomization_strategies import Strategies
import src.random_sampling as rand
from utils.generate_figures import generateFigures
from utils.generate_tables import writeLatexTables

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
    n_observations = 1000
    alpha = 2 * np.pi
    noise_level = 0.05
    regularization = 20
    random_vector_generator = rand.scaledIdentityCovGenerator()
    # random_vector_generator = rand.achlioptasRandomGenerator()
    # random_vector_generator = rand.rademacherRandomGenerator()
    # random_vector_generator = rand.exponentialRandomGenerator()
    
    observation_coords, observations = generateObservations(n_observations, alpha)
    true_parameter = trueParameter(alpha, observation_coords)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    noise_std = noise_level * np.amax(observations)
    forward_map = buildForwardMatrix(n_observations)
    
    solver_type = 'direct'
    problem_name = '1D_Deconvolution_Gaussian'

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(observation_coords, true_parameter)
    ax.set_title("True parameter")
    fig.tight_layout()
    plt.savefig(f"figures/{problem_name}/true_parameter.pdf", pad_inches=0, dpi=300)
    plt.close(fig)
    
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
        Strategies.ENKF_U1,
        Strategies.RSLS,
        Strategies.ALL
    ]

    # we want to perform fewer CG iterations on these randomization strategies
    # since the rank of the forward map is at most n_samples
    ls_strategies = [Strategies.RMA, Strategies.RMA_RMAP, Strategies.ALL]
    results = {}

    for curr_strategy in test_strategies:
        rand_solutions = []
        rand_labels = []
        for samples in n_random_samples:
            rand_labels.append(f"N = {samples}")
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_std**2), 0, regularization, random_vector_generator, samples, solver_type)
            
            # don't allow iterative solver to solve past rank of operator
            if solver_type == 'cg' and curr_strategy in ls_strategies:
                randomized_solver.solver.maxiter = samples
            else:
                # default
                randomized_solver.solver.maxiter = None
            rand_solutions.append(randomized_solver.solve())
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
                        f"figures/{problem_name}/{randomized_solver.name}.pdf", lims={"ylim": [-1.75, 1.75]})

    writeLatexTables(results, f'{problem_name}_table.tex')
    
