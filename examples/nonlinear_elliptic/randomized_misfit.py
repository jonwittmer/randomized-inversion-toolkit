import warnings
import numpy as np
import matplotlib.pyplot as plt
import ufl
import dolfin as dl
from hippylib import *  # from the tutorial

import logging
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

class PointwiseStateObservationRandomized(PointwiseStateObservation):
    def __init__(self, Vh, obs_points, n_random_vectors):
        """
        Constructor:
            :code:`Vh` is the finite element space for the state variable
            
            :code:`obs_points` is a 2D array number of points by geometric dimensions that stores \
            the location of the observations.
            
            :code:`n_random_vectors` is the number of random vectors to draw. The dimension of the \
            random vectors will be determined by the dimension of the observations.
        """
        self.B = assemblePointwiseObservation(Vh, obs_points)
        self.d = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.d, 0)
        self.Bu = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = None
        self.n_random_vectors = n_random_vectors
        self.random_vector_generator = None
        self.random_vectors = None        
        self.reduced_data = dl.Vector(self.B.mpi_comm(), n_random_vectors)

    def generate_random_vectors(self):
        # stored in rows so that the output of random_vectors @ data has size n_random_vectors
        if self.random_vector_generator is None:
            raise ValueError("random_vector_generator must be specified")
        self.random_vectors = self.random_vector_generator(self.n_random_vectors, self.d.local_size())
        return
    
    def cost(self, x):
        if self.random_vectors is None: 
            warnings.warn("Random vectors not generated yet. Generating random vectors.")
            self.generate_random_vectors()
        self.B.mult(x[STATE], self.Bu)
        # How do we insert randomized multiplication here?
        self.Bu.axpy(-1., self.d)
        self.reduced_data.set_local(self.random_vectors @ self.Bu.get_local())
        return 0.5*self.reduced_data.inner(self.reduced_data)
    
    def grad(self, i, x, out):
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            self.Bu.axpy(-1.0, self.d)
            # random sketching - it would be nice to sketch B from the left, then do the 
            # multiplication, but such a sketching may ruin the sparsity pattern of B
            # and we would need to store an additional matrix. This way works with the 
            # data structures that we already have.
            self.Bu.set_local(self.random_vectors.T @ (self.random_vectors @ self.Bu.get_local()))
            self.B.transpmult(self.Bu, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
            
    def apply_ij(self, i, j, dir, out):
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            # sketching multiplication
            self.Bu.set_local(self.random_vectors.T @ (self.random_vectors @ self.Bu.get_local()))
            self.B.transpmult(self.Bu, out)
        else:
            out.zero()
