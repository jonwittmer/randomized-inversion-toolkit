import argparse
import sys
import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from hippylib import *
from randomized_misfit import PointwiseStateObservationRandomized
from randomized_priors import BiLaplacianPriorRand
import colormaps as cm 

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=25)

# randomization strategy
class Strategy:
    NONE = "no_randomization"
    RMA = "rma"
    RMAP = "rmap"
    RMA_RMAP = "rma_rmap"
    RS = "rs"
    ENKF = "enkf"

# generate true parameter by drawing from prior
def priorSample(prior):
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    seed = 25
    parRandom = Random(dl.MPI.rank(dl.MPI.comm_world), dl.MPI.size(dl.MPI.comm_world), seed)
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue
    
# accept command line arguments so we don't have to keep changing script
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', help='what type of randomization strategy to use')
parser.add_argument('--n_random_vectors', help='number of random vectors')
args = parser.parse_args()
strategy = args.strategy
n_random_vectors = int(args.n_random_vectors)
print("Strategy: {}  n_random_vectors: {}".format(strategy, n_random_vectors))

# stuff for plotting
default_figsize = (7,5)
my_cmap = cm.make_cmap('div2-gray-gold.xml')
matplotlib.cm.register_cmap(cmap=my_cmap)
param_vmin = -1.0
param_vmax = 1.5

# problem parameters
ntargets = 100
rel_noise = 0.01
ndim = 2
nx = 128
ny = 128
mesh = dl.UnitSquareMesh(nx, ny)
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]
print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
    Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

# set up forward problem
def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)
u_bdr = dl.Expression("x[1]", degree=1)
u_bdr0 = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
f = dl.Constant(0.0)
def pde_varf(u,m,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

# set up prior
gamma = .1
delta = .5
theta0 = 2.
theta1 = .5
alpha  = np.pi/4
anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
anis_diff.set(theta0, theta1, alpha)
if strategy == Strategy.RS or strategy == Strategy.ENFK:
    local_size = Vh[PARAMETER].dim()
    random_vectors = 1. / (n_random_vectors)**(0.5) * np.random.normal(0, 1.0, (n_random_vectors, local_size))
    prior = BiLaplacianPriorRand(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True, random_vectors=random_vectors)
else:
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)

# generate true parameter as sample from prior
mtrue = priorSample(prior)
print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2))                
objs = dl.Function(Vh[PARAMETER],mtrue)
mytitle = "True Parameter"
plt.figure(figsize=default_figsize)
nb.plot(objs, mytitle=mytitle, cmap=my_cmap)
plt.savefig('figures/true_parameter.png')

# observations only on the bottom
targets_x = np.random.uniform(0.1,0.9, [ntargets] )
targets_y = np.random.uniform(0.1,0.5, [ntargets] )
targets = np.zeros([ntargets, ndim])
targets[:,0] = targets_x
targets[:,1] = targets_y
print( "Number of observation points: {0}".format(ntargets) )

# random number generator with covariance specified
if strategy == Strategy.RMA: 
    misfit = PointwiseStateObservationRandomized(Vh[STATE], targets, n_random_vectors)
else: # strategy == Strategy.NONE:
    misfit = PointwiseStateObservation(Vh[STATE], targets)

utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(x[STATE], x)
misfit.B.mult(x[STATE], misfit.d)
MAX = misfit.d.norm("linf")
noise_std_dev = rel_noise * MAX
parRandom.normal_perturb(noise_std_dev, misfit.d)

# set the random vector generator for randomized misfit class
if strategy == Strategy.RMA:
    def random_vector_generator(n_random_vectors, local_size):
        return 1. / (n_random_vectors)**(0.5) * np.random.normal(0, 1. / noise_std_dev, (n_random_vectors, local_size))
    misfit.random_vector_generator = random_vector_generator
    misfit.generate_random_vectors()
    
misfit.noise_variance = noise_std_dev**2
vmax = max( utrue.max(), misfit.d.max() )
vmin = min( utrue.min(), misfit.d.min() )

# save figures of state and observations
plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[STATE], utrue), mytitle="True State", vmin=vmin, vmax=vmax, cmap=my_cmap)
plt.savefig("figures/true_state.png")
plt.figure(figsize=default_figsize)
nb.plot_pts(targets, misfit.d, mytitle="Observations", vmin=vmin, vmax=vmax, cmap=my_cmap)
plt.savefig("figures/observations.png")

# create model
model = Model(pde, prior, misfit)

# solver parameters
m = prior.mean.copy()
compute_cost = prior.cost(m)
id_cost = prior.implicit_prior.cost(m)
print("compute cost: {}".format(compute_cost))
print("inf dim cost: {}".format(id_cost))

solver = ReducedSpaceNewtonCG(model)
solver.parameters["rel_tolerance"] = 1e-6
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"]      = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4

# solve inverse problem
x = solver.solve([None, m, None])
    
if solver.converged:
    print( "\nConverged in ", solver.it, " iterations.")
else:
    print( "\nNot Converged")
print( "Termination reason: ", solver.termination_reasons[solver.reason] )
print( "Final gradient norm: ", solver.final_grad_norm )
print( "Final cost: ", solver.final_cost )

plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[STATE], x[STATE]), mytitle="State", cmap=my_cmap)
plt.savefig("figures/final_state_{}_{}.png".format(strategy, n_random_vectors))
plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[PARAMETER], x[PARAMETER]), mytitle="Parameter", cmap=my_cmap, vmin=param_vmin, vmax=param_vmax)
plt.savefig("figures/final_parameter_{}_{}.png".format(strategy, n_random_vectors))
