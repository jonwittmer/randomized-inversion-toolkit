import dolfin as dl
import ufl
import numpy as np
import scipy.linalg as scila
import math

from hippylib.algorithms.linalg import MatMatMult, get_diagonal, amg_method, estimate_diagonal_inv2, Solver2Operator, Operator2Solver
from hippylib.algorithms.linSolvers import PETScKrylovSolver
from hippylib.algorithms.traceEstimator import TraceEstimator
from hippylib.algorithms.multivector import MultiVector
from hippylib.algorithms.randomizedEigensolver import doublePass, doublePassG

from hippylib.utils.random import parRandom
from hippylib.utils.vector2function import vector2Function
from hippylib.modeling.expression import ExpressionModule
from hippylib.modeling.prior import _Prior, _BilaplacianR, _BilaplacianRsolver

class nonSquareM(dl.LinearOperator):
    def __init__(self, shape, mpi_comm):
        super().__init__()
        self.shape = shape
        self.M = np.random.normal(0, 1, shape)
        self._mpi_comm = mpi_comm

    def mult(self, in_vec, out_vec):
        out_vec.set_local(self.M @ in_vec.get_local)
        
    def transpmult(self, in_vec, out_vec):
        out_vec.set_local(self.M.T @ in_vec.get_local)

    def size(self, dim):
        return self.shape[dim]

    def mpi_comm(self):
        return self._mpi_comm

    def init_vector(self, vector, dim):
        vector.init(self.mpi_comm(), self.size(dim))
        
        
class _BilaplacianMOperator(dl.LinearOperator):
    def __init__(self, sqrtM, random_vectors):
        self.sqrtM = sqrtM
        self.random_vectors = random_vectors
        self.help1 = dl.Vector(self.sqrtM.mpi_comm())
        self.help2 = dl.Vector(self.sqrtM.mpi_comm())
        self.sqrtM.init_vector(self.help1, 1)
        self.sqrtM.init_vector(self.help2, 1)
        print("help1 size: {}".format(self.help1.get_local().shape))
        print("help2 size: {}".format(self.help2.get_local().shape))
        print("random vectors shape: {}".format(self.random_vectors.shape))
        
        super().__init__(self.help1, self.help2)

    def mult(self, x, out):
        # TODO - VERIFY THIS IS CORRECT!!! NOT SURE ABOUT TRANSPOSE PART
        self.sqrtM.transpmult(x, self.help1)
        self.help2.set_local(self.random_vectors.T @ (self.random_vectors @ self.help1.get_local()))
        self.sqrtM.mult(self.help1, out)

    def init_vector(self, vec, dim):
        self.sqrtM.init_vector(vec, 0)

# minimal reproducible sample of trying to create a randomized bilaplacian prior
# currently the code crashes with trying to apply Msolver, so reproduce that error
mesh = dl.UnitSquareMesh(128, 128)
Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

trial = dl.TrialFunction(Vh)
test = dl.TestFunction(Vh)
varfM = ufl.inner(trial, test)*ufl.dx
M = dl.assemble(varfM)
test_vec = dl.Vector()
out_vec = dl.Vector()
M.init_vector(test_vec, 1)
M.init_vector(out_vec, 0)
random_vecs = np.random.normal(0., 1., (10, test_vec.size()))

# perhaps the stuff that is causing problems
# old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
# dl.parameters["form_compiler"]["quadrature_degree"] = -1
# qdegree = 2*Vh._ufl_element.degree()
# metadata = {"quadrature_degree" : qdegree}
# element = ufl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
# Qh = dl.FunctionSpace(Vh.mesh(), element)    
# ph = dl.TrialFunction(Qh)
# qh = dl.TestFunction(Qh)
# Mqh = dl.assemble(ufl.inner(ph,qh)*ufl.dx(metadata=metadata))
# one_constant = dl.Constant(1.)
# ones = dl.interpolate(one_constant, Qh).vector()
# dMqh = Mqh*ones
# Mqh.zero()
# dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
# Mqh.set_diagonal(dMqh)
# MixedM = dl.assemble(ufl.inner(ph,test)*ufl.dx(metadata=metadata))
# sqrtM = MatMatMult(MixedM, Mqh)

M_ns = nonSquareM((16641, 98304), M.mpi_comm())
print(M_ns.size(0), M_ns.size(1))
M_operator = _BilaplacianMOperator(M, random_vecs)

Msolver = PETScKrylovSolver(Vh.mesh().mpi_comm(), "cg", "none")
Msolver.set_operator(M_operator)
Msolver.parameters["maximum_iterations"] = 25
Msolver.parameters["relative_tolerance"] = 1e-6
Msolver.parameters["absolute_tolerance"] = 1e-10
Msolver.parameters["error_on_nonconvergence"] = True
Msolver.parameters["nonzero_initial_guess"] = False

print(out_vec.get_local())
test_vec.set_local(np.random.normal(0, 1, test_vec.size()))
M_operator.mult(test_vec, out_vec)
print(out_vec.get_local())

Msolver.solve(test_vec, out_vec)




