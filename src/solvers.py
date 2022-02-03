import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

class Solver:
    def __init__(self):
        pass
        
    def solve(self):
        raise NotImplementedError("Solver base class does not implement solve method")

class NumpyDirectSolver(Solver):
    def __init__(self):
        self.name = 'numpy least squares'
        self.solver_type = 'direct'
    
    def solve(self, A, b):
        output = np.linalg.lstsq(A, b)
        return output[0]

class CgSolver(Solver):
    def __init__(self):
        self.name = 'cg'
        self.solver_type = 'iterative'
        self.atol = None
        self.tol = 1e-5
        self.maxiter = None
        
    # A can be a matrix or a LinearOperator
    def solve(self, A, b):
        output = sp.sparse.linalg.cg(A, b, atol=self.atol, tol=self.tol, maxiter=self.maxiter)
        return output[0]
        
        
