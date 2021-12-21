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
        pass
    
    def solve(self, A, b):
        return np.linalg.lstsq(A, b)

class CgSolver(Solver):
    def __init__(self):
        pass

    # A can be a matrix or a LinearOperator
    def solve(self, A, b):
        return sc.sparse.linalg.cg(A, b)
        
        
