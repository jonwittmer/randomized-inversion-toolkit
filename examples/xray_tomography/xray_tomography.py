import os
import sys
sys.path.append('../../')

import numpy as np
import scipy as sp
import scipy.sparse.linalg as sp_sparse
import skimage.data as sk_data
import skimage.transform as sk_transform

from src.randomization_strategies import Strategies
from utils.generate_tables import writeLatexTables

def trueParameter(dimension):
    original = sk_data.shepp_logan_phantom()
    parameter = sk_transform.resize(original, (dimension, dimension), anti_aliasing=True)
    return parameter

def generateObservations(true_parameter, n_observations):
    projection_angles = np.linspace(0, 180, n_observations)
    sinogram = sk_transform.radon(true_parameter, projection_angles)
    return sinogram

def buildForwardOperator(parameter_shape, observation_shape):
    projection_angles = np.linspace(0, 180, observation_shape[1])
    def radon_flat_image_only(flat_image):
        two_d_image = np.reshape(flat_image, parameter_shape)
        sinogram = sk_transform.radon(two_d_image, projection_angles)
        return sinogram.reshape((-1,))

    def iradon_flat_sinogram_only(flat_sinogram):
        sinogram = np.reshape(flat_sinogram, observation_shape)
        image = iradon(sinogram, projection_angles, filter_name=None)

        # correction factor adapted from https://archive.siam.org/books/cs10/Xray_MatrixFree/XRC_Tikhonov_comp.m
        # see https://www2.math.upenn.edu/~ccroke/chap6.pdf for more information on relationship between inverse transformation and adjoint
        correction_factor = 40.73499999 * parameter_shape[0] / 64
        image *= correction_factor
        return image.reshape((-1,))

    operator_shape = (observation_shape[0] * observation_shape[1], parameter_shape[0] * parameter_shape[1])
    forward_map = sp_sparse.LinearOperator(operator_shape, matvec=radon_flat_image_only, rmatvec=iradon_flat_sinogram_only)
    return forward_map

if __name__ == '__main__':
    np.random.seed(20)
    
    # problem setup
    n_angles = 180
    image_dimension = 64
    noise_level = 0.01
    regularization = 200 # we will use identiy prior_covariance, parameterized by scalar given here
    random_vector_generator = np.random.multivariate_normal
    
    true_parameter = trueParameter(image_dimension)
    observations = generateObservations(true_parameter, n_angles)
    forward_map = buildForwardOperator(true_parameter.shape, observations.shape)
    data = observations + np.amax(observations) * np.random.normal(0, noise_level, observations.shape)
    data = data.reshape((-1,))
    
    solver_type = 'cg'
    problem_name = 'xray_tomography'

    # generate u1 solution only once
    no_randomizaton_solver = Strategies.NO_RANDOMIZATION(data, forward_map, 1 / (noise_level**2), 0, regularization, random_vector_generator, 0, solver_type)
    u1_solution = no_randomizaton_solver.solve().reshape(true_parameter.shape)
    
    n_random_samples = [10, 100, 1000]
    test_strategies = [
        #Strategies.RMAP,
        Strategies.RMA,
        #Strategies.RMA_RMAP,
        #Strategies.RS_U1,
        #Strategies.RS,
        #Strategies.ENKF,
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
            results[randomized_solver.name]["samples"].append(samples.reshape(true_parameter.shape))
            results[randomized_solver.name]["rel_error"].append(np.linalg.norm(rand_solutions[-1] - u1_solution) / np.linalg.norm(u1_solution))
            print(f"N = {samples}    error = {results[randomized_solver.name]['rel_error'][-1]}")
        print()
        #generateFigures(observation_coords, u1_solution, rand_solutions, rand_labels,
        #                f"figures/{problem_name}/{randomized_solver.name}.png", lims={"ylim": [-2, 2]})

    writeLatexTables(results, f'{problem_name}_table.tex')
