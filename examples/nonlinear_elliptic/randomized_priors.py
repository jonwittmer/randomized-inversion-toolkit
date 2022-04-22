# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import ufl
import numpy as np
import scipy.linalg as scila
import scipy.sparse as scisp

from hippylib.modeling.prior import _Prior, BiLaplacianPrior
from hippylib.algorithms.linSolvers import PETScKrylovSolver
from hippylib.algorithms.linalg import amg_method

def fenics_operator_to_numpy(comm, operator_mult_function, operator_shape):
    # build row-by-row by acting on unit vectors
    numpy_vector = np.zeros((operator_shape[0],))
    numpy_mat = np.zeros((operator_shape))
    unit_vector = dl.Vector(comm, operator_shape[0])
    row = dl.Vector(comm, operator_shape[0])
    for i in range(operator_shape[0]):
        # set the unit vector to be the ith basis vector
        numpy_vector[i] = 1
        numpy_vector[(i-1) % operator_shape[0]] = 0
        unit_vector.set_local(numpy_vector)
        operator_mult_function(unit_vector, row)
        numpy_row = row.get_local()
        numpy_mat[i, :] = numpy_row
    return numpy_mat

class ROperator(dl.LinearOperator):
    def __init__(self, R_half, random_vectors, mpi_comm):
        self.help_vector = dl.Vector()
        self.help_vector.init(R_half.shape[0])
        super().__init__(self.help_vector, self.help_vector)
        self.R_half = R_half
        self.random_vectors = random_vectors
        self._mpi_comm = mpi_comm

    def mult(self, in_vec, out_vec):
        # this operator represents a square matrix
        if out_vec.size() == 0:
            out_vec.init(in_vec.size())
        out_vec.set_local(self.R_half @ (self.random_vectors.T @ (self.random_vectors @ (self.R_half.T @ in_vec.get_local()))))
        
    def mpi_comm(self):
        return self._mpi_comm

class IdentityPreconditioner:
    def solve(self, out_vec, in_vec):
        out_vec.set_local(in_vec.get_local())
        
class BiLaplacianPriorRand(_Prior):
    """
    This prior is not scalable to multiple processors as it assembles the full prior inverse covariance matrix
    """
    def __init__(self, Vh, gamma, delta, Theta=None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False, random_vectors=None):
        self.random_vectors = random_vectors
        self.implicit_prior = BiLaplacianPrior(Vh, gamma, delta, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False)
        self.prior_dim = (Vh.dim(), Vh.dim())
        self.mean = mean
        if self.mean is None:
            self.mean = dl.Vector(self.implicit_prior.R.mpi_comm(), self.prior_dim[1])
        self.constructCholeskyOfMinv()
        self.R = ROperator(self.R_half, self.random_vectors, self.implicit_prior.R.mpi_comm())
        #self.Rsolver = IdentityPreconditioner()

        # since Rsolver is only used for CG preconditioning, it is alright to use non-randomized preconditioner
        # randomized  matrix is not invertible anyways so cannot be used as a preconditioner
        self.Rsolver = self.implicit_prior.Rsolver
        
        # this is used only for evaluating the norm of the gradient in the Mass weighted inner product,
        # so we don't need to randomize it. Msolver is not used in application of prior covariance
        self.Msolver = self.implicit_prior.Msolver
        
    def constructCholeskyOfMinv(self):
        print("Constructing matrix representation of prior")

        # solving as matrix multiplication
        matrix_representation = fenics_operator_to_numpy(self.implicit_prior.M.mpi_comm(), self.implicit_prior.R.mult, self.prior_dim)

        print("Starting cholesky factorization")
        self.R_half = np.linalg.cholesky(matrix_representation)
        # check whether cholesky is done right
        diff = matrix_representation - self.R_half @ self.R_half.T
        if np.linalg.norm(diff, np.inf) > 1e-6:
            raise ValueError("Cholesky factorization failed - error = {}".format(np.linalg.norm(diff, np.inf)) )
        
    # this function is not needed, but it makes the API consistent with a linear operator
    def size(self, dim):
        return self.prior_dim(dim)

    def init_vector(self, x, dim):
        if dim == "noise":
            self.implicit_prior.init_vector(x, dim)
        else:
            x.init(self.prior_dim[0])

    def sample(self, noise, s, add_mean=True):
        self.implicit_prior.sample(noise, s, add_mean)
