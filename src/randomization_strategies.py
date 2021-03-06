import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from .solvers import Solver, NumpyDirectSolver, CgSolver

str_to_solver_mapping = {
    'direct' : NumpyDirectSolver(),
    'cg' : CgSolver()
}

def isScalar(val):
    if (val * np.array([1])).size == 1:
        return True
    else:
        return False

class RandomizationStrategy:
    data = None
    forward_map = None
    inv_noise_covariance = None
    prior_mean = None
    inv_prior_covariance = None
    
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        # random_vector_generator should take arguments(mean, covariance, shape)
        self.data = data
        self.forward_map = forward_map
        self.inv_noise_covariance = inv_noise_covariance
        self.prior_mean = prior_mean
        self.inv_prior_covariance = inv_prior_covariance
        self.random_vector_generator = random_vector_generator
        self.n_random_samples = n_random_samples
        self.parameter_dim = forward_map.shape[1]
        
        self.solver = self.getSolver(solver)
        print('using {} as solver'.format(self.solver.name))

        self.reshapePriorMean()
        self.hessian = None

        # some methods require the covariances in addition to or instead of the inverses
        # set to None for now, the methods will implement these matrices if needed
        self.noise_covariance = None
        self.prior_covariance = None
        

    def getCovarianceFromInverse(self, inv_covariance):
        # this function will not be called if # to avoid calling np.linalg.inv().
        if isScalar(inv_covariance):
            return 1 / inv_covariance
        else:
            return np.linalg.inv(self.inv_noise_covariance)
        
    def getSolver(self, solver):
        if isinstance(solver, str):
            if solver in str_to_solver_mapping:
                return str_to_solver_mapping[solver]
            else:
                raise ValueError("{} not in str_to_solver_mapping. Options are: ({})".format(solver, str_to_solver_mapping.keys()))
        elif isinstance(solver, Solver):
            return solver
        else:
            raise ValueError("solver must either be a string or Solver instance. String options are: {}".format(str_to_solver_mapping.keys()))

    def reshapePriorMean(self):
        # convert scalar to vector of constants, always reshape to ensure correct dimensions
        if isScalar(self.prior_mean):
            self.prior_mean = self.prior_mean * np.ones((self.parameter_dim,))
        self.prior_mean = np.reshape(self.prior_mean, (self.parameter_dim,))

    def convertCovariancesToMatrices(self):
        if isScalar(self.inv_prior_covariance):
            self.inv_prior_covariance = self.inv_prior_covariance * np.identity(self.parameter_dim)
        if isScalar(self.inv_noise_covariance):
            self.inv_noise_covariance = self.inv_noise_covariance * np.identity(self.data.shape[0])

        # not all methods have these
        if self.noise_covariance is not None:
            if isScalar(self.noise_covariance):
                self.noise_covariance = self.noise_covariance * np.identity(self.data.shape[0])
        if self.prior_covariance is not None:
            if isScalar(self.prior_covariance):
                self.prior_covariance = self.prior_covariance * np.identity(self.parameter_dim)
        
    def solve(self):
        raise NotImplementedError("RandomizationStrategy base class does not implement solve() function")

    
class NoRandomizationStrategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.name = 'no\_rand'
        
    def formHessian(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def hessianAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map @ v
                temp = self.inv_noise_covariance @ temp
                temp = self.forward_map.transpose() @ temp
                return temp + self.inv_prior_covariance @ v
            self.hessian = sp.sparse.linalg.LinearOperator((self.parameter_dim, self.parameter_dim), hessianAction)
        else:
            self.hessian = self.forward_map.T @ self.inv_noise_covariance @ self.forward_map + self.inv_prior_covariance
        
    def solve(self):
        if isScalar(self.inv_prior_covariance):
            self.inv_prior_covariance = self.inv_prior_covariance * np.identity(self.parameter_dim)
        if isScalar(self.inv_noise_covariance):
            self.inv_noise_covariance = self.inv_noise_covariance * np.identity(self.parameter_dim)
            
        # form right hand side without storing intermediate matrices (only matvec products)
        rhs = self.inv_noise_covariance @ self.data
        rhs = self.forward_map.transpose() @ rhs + self.inv_prior_covariance @ self.prior_mean

        # form hessian if not already formed
        if self.hessian is None:
            self.formHessian()
            
        return self.solver.solve(self.hessian, rhs)

    
class LeftSketchingStrategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.name = 'RMA'

    def formHessian(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def hessianAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map @ v
                temp = self.projection_vectors @ (self.projection_vectors.T @ temp)
                temp = self.forward_map.transpose() @ temp
                return temp + self.inv_prior_covariance @ v
            self.hessian = sp.sparse.linalg.LinearOperator((self.parameter_dim, self.parameter_dim), hessianAction)
        else:
            self.hessian = self.forward_map.T @ self.projection_vectors @ self.projection_vectors.T @ self.forward_map + self.inv_prior_covariance
        
    def drawRandomVectors(self):            
        self.projection_vectors = self.random_vector_generator(np.zeros_like(self.data), self.inv_noise_covariance, self.n_random_samples).T
        self.projection_vectors *= 1 / (self.n_random_samples**0.5)
        
    def solve(self):
        self.convertCovariancesToMatrices()
        self.drawRandomVectors()
        
        # form hessian if not already formed
        if self.hessian is None:
            self.formHessian()
            
        # form right hand side without storing intermediate matrices (only matvec products)
        rhs = self.projection_vectors @ (self.projection_vectors.T @ self.data)
        rhs = self.forward_map.transpose() @ rhs + self.inv_prior_covariance @ self.prior_mean
            
        return self.solver.solve(self.hessian, rhs)

    
class RmapStrategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.name = 'RMAP'
        
    def formHessian(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def hessianAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map @ v
                temp = self.inv_noise_covariance @ temp
                temp = self.forward_map.transpose() @ temp
                return temp + self.inv_prior_covariance @ v
            self.hessian = sp.sparse.linalg.LinearOperator((self.parameter_dim, self.parameter_dim), hessianAction)
        else:
            self.hessian = self.forward_map.T @ self.inv_noise_covariance @ self.forward_map + self.inv_prior_covariance

    def solveRealization(self, data_perturbation, prior_perturbation):
        # form hessian if not already formed
        if self.hessian is None:
            self.formHessian()
            
        # form right hand side without storing intermediate matrices (only matvec products)
        rhs = self.inv_noise_covariance @ (self.data + data_perturbation)
        rhs = self.forward_map.transpose() @ rhs + self.inv_prior_covariance @ (self.prior_mean + prior_perturbation)
            
        return self.solver.solve(self.hessian, rhs)
        
    def solve(self):
        if self.noise_covariance is None:
            self.noise_covariance = self.getCovarianceFromInverse(self.inv_noise_covariance)
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.convertCovariancesToMatrices()
        
        # allocate storage for all realizations
        results = np.zeros((self.parameter_dim, self.n_random_samples))
        for i in range(self.n_random_samples):
            data_perturbation = self.random_vector_generator(np.zeros_like(self.data), self.noise_covariance, 1).reshape(self.data.shape)
            prior_perturbation = self.random_vector_generator(np.zeros_like(self.prior_mean), self.prior_covariance, 1).reshape(self.prior_mean.shape)
            results[:, i] = np.reshape(self.solveRealization(data_perturbation, prior_perturbation), (self.parameter_dim,))
        return np.mean(results, axis=1)


class LeftSketchingRmapStrategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.name = 'RMA+RMAP'
        
    def formHessian(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def hessianAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map @ v
                temp = self.projection_vectors @ (self.projection_vectors.T @ temp)
                temp = self.forward_map.transpose() @ temp
                return temp + self.inv_prior_covariance @ v
            self.hessian = sp.sparse.linalg.LinearOperator((self.parameter_dim, self.parameter_dim), hessianAction)
        else:
            self.hessian = self.forward_map.T @ self.projection_vectors @ self.projection_vectors.T @ self.forward_map + self.inv_prior_covariance
        
    def drawRandomVectors(self):
        self.projection_vectors = self.random_vector_generator(np.zeros_like(self.data), self.inv_noise_covariance, self.n_random_samples).T
        self.projection_vectors *= 1 / (self.n_random_samples**0.5)
        
    def solveRealization(self, data_perturbation, prior_perturbation):
        # form hessian if not already formed
        if self.hessian is None:
            self.formHessian()
            
        # form right hand side without storing intermediate matrices (only matvec products)
        rhs = self.projection_vectors @ (self.projection_vectors.T @ (self.data + data_perturbation))
        rhs = self.forward_map.transpose() @ rhs + self.inv_prior_covariance @ (self.prior_mean + prior_perturbation)
            
        return self.solver.solve(self.hessian, rhs)

    def solve(self):
        # for sampling, we need the covariances, not the inverses for this method
        if self.noise_covariance is None:
            self.noise_covariance = self.getCovarianceFromInverse(self.inv_noise_covariance)
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.convertCovariancesToMatrices()
        self.drawRandomVectors()
        
        # allocate storage for all realizations
        results = np.zeros((self.parameter_dim, self.n_random_samples))
        for i in range(self.n_random_samples):
            data_perturbation = self.random_vector_generator(np.zeros_like(self.data), self.noise_covariance, 1).reshape(self.data.shape)
            prior_perturbation = self.random_vector_generator(np.zeros_like(self.prior_mean), self.prior_covariance, 1).reshape(self.prior_mean.shape)
            results[:, i] = np.reshape(self.solveRealization(data_perturbation, prior_perturbation), (self.parameter_dim,))
        return np.mean(results, axis=1)
    
    
class RightSketchU1Strategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.name = 'RS\_U1'
        
    def formHessian(self):            
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def hessianAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map @ v
                temp = self.inv_noise_covariance @ temp
                temp = self.forward_map.transpose() @ temp
                return temp + self.projection_vectors @ (self.projection_vectors.T @ v)
            self.hessian = sp.sparse.linalg.LinearOperator((self.parameter_dim, self.parameter_dim), hessianAction)
        else:
            self.hessian = self.forward_map.T @ self.inv_noise_covariance @ self.forward_map + self.projection_vectors @ self.projection_vectors.T
        
    def drawRandomVectors(self):
        self.projection_vectors = self.random_vector_generator(np.zeros_like(self.prior_mean), self.inv_prior_covariance, self.n_random_samples).T
        self.projection_vectors *= 1 / (self.n_random_samples**0.5)
        
    def solve(self):
        self.convertCovariancesToMatrices()
        self.drawRandomVectors()
        
        # form hessian if not already formed
        if self.hessian is None:
            self.formHessian()
                
        # form right hand side without storing intermediate matrices (only matvec products)
        rhs = self.inv_noise_covariance @ self.data
        rhs = self.forward_map.transpose() @ rhs + self.projection_vectors @ (self.projection_vectors.T @ self.prior_mean)
            
        return self.solver.solve(self.hessian, rhs)

    
class RightSketchU2Strategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.innovation = None
        self.name = 'RS'

    def drawRandomVectors(self):
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.projection_vectors = self.random_vector_generator(np.zeros_like(self.prior_mean), self.prior_covariance, self.n_random_samples).T
        self.projection_vectors *= 1 / (self.n_random_samples**0.5)

    def formInnovation(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def innovationAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map.transpose() @ v
                temp = self.projection_vectors @ (self.projection_vectors.T @ temp)
                temp = self.forward_map @ temp
                return temp + self.noise_covariance @ v
            self.innovation = sp.sparse.linalg.LinearOperator((self.data.shape[0], self.data.shape[0]), innovationAction)
        else:
            self.innovation = self.forward_map @ self.projection_vectors @ self.projection_vectors.T @ self.forward_map.T + self.noise_covariance
        
    def solve(self):
        if self.noise_covariance is None:
            self.noise_covariance = self.getCovarianceFromInverse(self.inv_noise_covariance)
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.convertCovariancesToMatrices()
        self.drawRandomVectors()        
                
        # innovation name comes from Kalman filtering literature
        if self.innovation is None:
            self.formInnovation()
            
        rhs = self.data - self.forward_map @ self.prior_mean
        temp = self.solver.solve(self.innovation, rhs)
        return self.prior_mean + self.projection_vectors @ (self.projection_vectors.T @ (self.forward_map.transpose() @ temp))

class EnkfStrategy(RandomizationStrategy):
    def __init__(self, data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver):
        super().__init__(data, forward_map, inv_noise_covariance, prior_mean, inv_prior_covariance, random_vector_generator, n_random_samples, solver)
        self.innovation = None
        self.name = 'ENKF'

    def drawRandomVectors(self):
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.projection_vectors = self.random_vector_generator(np.zeros_like(self.prior_mean), self.prior_covariance, self.n_random_samples).T
        self.projection_vectors *= 1 / (self.n_random_samples**0.5)

    def formInnovation(self):
        if isinstance(self.forward_map, sp.sparse.linalg.LinearOperator):
            def innovationAction(v):
                # compute (A^T * inv_noise_cov * A + inv_prior_cov) * v
                temp = self.forward_map.transpose() @ v
                temp = self.projection_vectors @ (self.projection_vectors.T @ temp)
                temp = self.forward_map @ temp
                return temp + self.noise_covariance @ v
            self.innovation = sp.sparse.linalg.LinearOperator((self.data.shape[0], self.data.shape[0]), innovationAction)
        else:
            self.innovation = self.forward_map @ self.projection_vectors @ self.projection_vectors.T @ self.forward_map.T + self.noise_covariance
        
    def solveRealization(self, data_perturbation, prior_perturbation):
        # innovation name comes from Kalman filtering literature
        if self.innovation is None:
            self.formInnovation()
            
        rhs = self.data + data_perturbation - self.forward_map @ (self.prior_mean + prior_perturbation)
        temp = self.solver.solve(self.innovation, rhs)
        return self.prior_mean +  self.projection_vectors @ (self.projection_vectors.T @ (self.forward_map.transpose() @ temp))

    # allocate storage for all realizations
    def solve(self):
        if self.noise_covariance is None:
            self.noise_covariance = self.getCovarianceFromInverse(self.inv_noise_covariance)
        if self.prior_covariance is None:
            self.prior_covariance = self.getCovarianceFromInverse(self.inv_prior_covariance)
        self.convertCovariancesToMatrices()
        self.drawRandomVectors()
        
        results = np.zeros((self.parameter_dim, self.n_random_samples))
        for i in range(self.n_random_samples):
            data_perturbation = self.random_vector_generator(np.zeros_like(self.data), self.noise_covariance, 1).reshape(self.data.shape)
            prior_perturbation = self.random_vector_generator(np.zeros_like(self.prior_mean), self.prior_covariance, 1).reshape(self.prior_mean.shape)
            results[:, i] = np.reshape(self.solveRealization(data_perturbation, prior_perturbation), (self.parameter_dim,))
        return np.mean(results, axis=1)

class Strategies:
    NO_RANDOMIZATION = NoRandomizationStrategy
    RMA = LeftSketchingStrategy
    LS = RMA
    RMAP = RmapStrategy
    RMA_RMAP = LeftSketchingRmapStrategy
    RS = RightSketchU2Strategy
    RS_U1 = RightSketchU1Strategy
    ENKF = EnkfStrategy
