import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.stats as sp_stats
import scipy.sparse.linalg as sp_sparse
import skimage.data as sk_data
import skimage.transform as sk_transform
import matplotlib.pyplot as plt
import multiprocessing as mult_proc

import src.random_sampling as rand
from src.randomization_strategies import Strategies
from utils.generate_tables import writeLatexTables
from utils.generate_figures import checkDirectory
from utils.lcurve import computeLCurve

def trueParameter(dimension):
    original = sk_data.shepp_logan_phantom()
    parameter = sk_transform.resize(original, (dimension, dimension), anti_aliasing=True)
    return parameter

def generateObservations(true_parameter, n_observations):
    projection_angles = np.linspace(0, 180, n_observations)
    sinogram = sk_transform.radon(true_parameter, projection_angles, circle=False)
    return sinogram

_func = None
def bounce_init(func):
    global _func
    _func = func
def bounce(x):
    return _func(x)

def buildMat(linear_operator):
    mat = np.zeros(linear_operator.shape)
    def f(i):
        if (i % 100) == 0:
            print(i)
        e = np.zeros((linear_operator.shape[1],))
        e[i] = 1
        return linear_operator @ e
    
    with mult_proc.Pool(initializer=bounce_init, initargs=(f ,)) as pool:
        results = pool.map(bounce, range(linear_operator.shape[1]))
        for i in range(linear_operator.shape[1]):
            mat[:, i] = results[i]
    return mat

def buildForwardOperator(parameter_shape, observation_shape):
    projection_angles = np.linspace(0, 180, observation_shape[1])
    def radon_flat_image_only(flat_image):
        two_d_image = np.reshape(flat_image, parameter_shape)
        sinogram = sk_transform.radon(two_d_image, projection_angles, circle=False)
        return sinogram.reshape((-1,))

    def iradon_flat_sinogram_only(flat_sinogram):
        sinogram = np.reshape(flat_sinogram, observation_shape)
        image = sk_transform.iradon(sinogram, projection_angles, filter_name=None, circle=False)

        # correction factor adapted from https://archive.siam.org/books/cs10/Xray_MatrixFree/XRC_Tikhonov_comp.m
        # see https://www2.math.upenn.edu/~ccroke/chap6.pdf for more information on relationship between inverse transformation and adjoint
        correction_factor = 40.73499999 * parameter_shape[0] / 64
        image *= correction_factor
        return image.reshape((-1,))

    operator_shape = (observation_shape[0] * observation_shape[1], parameter_shape[0] * parameter_shape[1])
    forward_map = sp_sparse.LinearOperator(operator_shape, matvec=radon_flat_image_only, rmatvec=iradon_flat_sinogram_only)
    forward_map = buildMat(forward_map)
    return forward_map

def plotSpectrum(linear_operator):    
    # SVD
    print("performing SVD")
    u, evals, vh = np.linalg.svd(mat)
    
    fig, ax = plt.subplots(figsize=(3,5))
    ax.semilogy(evals, range(evals.shape[0]))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    plt.savefig("spectrum.pdf", pad_inches=0, dpi=300)

def saveSolution(solution, problem_name, method_name, n_samples):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(solution, vmin=0, vmax=1, cmap='gray')
    ax.axis('off')
    fig_name = f'figures/{problem_name}/{method_name}_N{n_samples:05d}.pdf'
    checkDirectory(fig_name)
    print(f'\tsaving figure to {fig_name}\n')
    plt.savefig(fig_name, pad_inches=0, dpi=300)
    plt.close(fig)

    # write raw data to file
    data_name = f'solutions/{problem_name}/{method_name}_N{n_samples:05d}.pdf'
    checkDirectory(data_name)
    np.save(data_name, solution)
    
if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_angles = 45
    image_dimension = 64
    noise_level = 0.01
    regularization = 2500 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = rand.scaledIdentityCovGenerator()
    
    true_parameter = trueParameter(image_dimension)
    observations = generateObservations(true_parameter, n_angles)
    forward_map = buildForwardOperator(true_parameter.shape, observations.shape)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    noise_std = noise_level * np.amax(observations)
    data = data.reshape((-1,))
    
    solver_type = 'cg'
    problem_name = 'xray_tomography_gaussian'

    #plotSpectrum(forward_map)
    
    def strategyBuilder(reg):
        return Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_std**2), 0, reg, random_vector_generator, 0, solver_type)
    #computeLCurve(true_parameter, strategyBuilder)
    
    # generate u1 solution only once
    no_randomization_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_std**2), 0, regularization, random_vector_generator, 0, solver_type)
    no_randomization_solver.solver.atol = 1e-5
    no_randomization_solver.solver.tol = 1e-5
    no_randomization_solver.solver.maxiter = 2000
    u1_solution = no_randomization_solver.solve().reshape(true_parameter.shape)
    saveSolution(u1_solution, problem_name, 'u1', 0)
    
    n_random_samples = [10, 100, 1000, 5000]
    test_strategies = [
        Strategies.RMAP,
        Strategies.RMA,
        Strategies.RMA_RMAP,
        #Strategies.RS_U1,
        Strategies.RS,
        Strategies.ENKF,
        #Strategies.RSLS,
        #Strategies.ENKF_U1,
        Strategies.ALL
    ]
    results = {}
    
    for curr_strategy in test_strategies:
        rand_solutions = []
        rand_labels = []
        for samples in n_random_samples:
            rand_labels.append(f"N = {samples}")
            randomized_solver = curr_strategy(data, forward_map, 1 / (noise_std**2), 0, regularization, random_vector_generator, samples, solver_type)
            print(f'solving with {randomized_solver.name}, N = {samples}')
            randomized_solver.solver.atol = 1e-5
            randomized_solver.solver.tol = 1e-5
            randomized_solver.solver.maxiter = 2000
            rand_solutions.append(randomized_solver.solve().reshape(true_parameter.shape))
            
            if randomized_solver.name not in results:
                results[randomized_solver.name] = {"samples": [], "rel_error": []}
            results[randomized_solver.name]["samples"].append(samples)
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"\tN = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")

            saveSolution(rand_solutions[-1], problem_name, randomized_solver.name, samples)
        print()
        
    #writeLatexTables(results, f'{problem_name}_table.tex')
