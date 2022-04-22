import argparse
import sys
import os
sys.path.insert(0, os.path.realpath('../..'))
import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from hippylib import *
from randomized_misfit import PointwiseStateObservationRandomized
from randomized_priors import BiLaplacianPriorRand, fenics_operator_to_numpy
import utils.colormaps as cm 

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
    RS_U1 = "rs"
    ENKF_U1 = "enkf"
    
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

def inversePriorSample(prior, R_half=None):
    sample = dl.Vector()
    prior.init_vector(sample, 0)
    parRandom.normal(1., sample)

    # make sure we have square root of prior inverse covariance
    if isinstance(prior, BiLaplacianPriorRand):
        R_half = prior.R_half
    elif R_half is None:
        print("Forming matrix representation of prior inverse covariance")
        matrix_representation = fenics_operator_to_numpy(prior.M.mpi_comm(), prior.R.mult, (sample.size(), sample.size()))
        R_half = np.linalg.cholesky(matrix_representation)
        
    sample.set_local(R_half @ sample.get_local())
    prior.mean.axpy(1.0, sample)
    return sample, R_half

def solveInstance(pde, prior, misfit):
    print("solving instance")
    
    # create inverse problem model
    model = Model(pde, prior, misfit)
    
    # solver parameters
    solver = ReducedSpaceNewtonCG(model)
    solver.parameters["rel_tolerance"] = 1e-6
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 25
    solver.parameters["GN_iter"] = 10
    solver.parameters["globalization"] = "LS"
    solver.parameters["LS"]["c_armijo"] = 1e-4
    
    # solve inverse problem
    m = prior.mean.copy()
    x = solver.solve([None, m, None])
    
    if solver.converged:
        print( "\nConverged in ", solver.it, " iterations.")
    else:
        print( "\nNot Converged")
    print( "Termination reason: ", solver.termination_reasons[solver.reason] )
    print( "Final gradient norm: ", solver.final_grad_norm )
    print( "Final cost: ", solver.final_cost )
    return x

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
my_cmap = cm.make_cmap('../../utils/div2-gray-gold.xml')
matplotlib.cm.register_cmap(cmap=my_cmap)
param_vmin = -1.0
param_vmax = 4

# problem parameters
ntargets = 100
rel_noise = 0.001
ndim = 2
nx = 32
ny = 32
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
gamma = 0.1
delta = 0.5
theta0 = 2.
theta1 = .5
alpha  = np.pi/4
anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree=1)
anis_diff.set(theta0, theta1, alpha)
if strategy == Strategy.RS_U1 or strategy == Strategy.ENKF_U1:
    local_size = Vh[PARAMETER].dim()
    random_vectors = 1. / (n_random_vectors)**(0.5) * np.random.normal(0, 1.0, (n_random_vectors, local_size))
    prior = BiLaplacianPriorRand(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True, random_vectors=random_vectors)
else:
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)
print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))

# generate true parameter as sample from prior
mtrue = priorSample(prior)
true_sol = mtrue.copy()
objs = dl.Function(Vh[PARAMETER], mtrue)
mytitle = "True Parameter"
fig = plt.figure(figsize=default_figsize)
nb.plot(objs, mytitle=mytitle, cmap=my_cmap, vmin=param_vmin, vmax=param_vmax)
fig.tight_layout()
plt.savefig('figures/true_parameter.png')

# observations only on the boundary
targets_x = np.random.uniform(0.1, 0.9, [ntargets] )
targets_y = np.random.uniform(0.0, 0.5, [ntargets] )
targets = np.zeros([ntargets, ndim])
targets[:,0] = targets_x
targets[:,1] = targets_y
print("Number of observation points: {0}".format(ntargets))

# misfit: || d - Bu ||^2 with observations d at targets
if strategy == Strategy.RMA or strategy == Strategy.RMA_RMAP: 
    misfit = PointwiseStateObservationRandomized(Vh[STATE], targets, n_random_vectors)
else:
    misfit = PointwiseStateObservation(Vh[STATE], targets)

# generate observations
utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(x[STATE], x)
misfit.B.mult(x[STATE], misfit.d)
observations = misfit.d.copy()
MAX = misfit.d.norm("linf")
noise_std_dev = rel_noise * MAX
parRandom.normal_perturb(noise_std_dev, misfit.d)

# set the random vector generator for randomized misfit class
if strategy == Strategy.RMA or strategy == Strategy.RMA_RMAP:
    def random_vector_generator(n_random_vectors, local_size):
        return 1. / (n_random_vectors)**(0.5) * np.random.normal(0, 1. / noise_std_dev, (n_random_vectors, local_size))
    misfit.random_vector_generator = random_vector_generator
    misfit.generate_random_vectors()
    
misfit.noise_variance = noise_std_dev**2
vmax = max( utrue.max(), misfit.d.max() )
vmin = min( utrue.min(), misfit.d.min() )

# save figures of state and observations
fig = plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[STATE], utrue), mytitle="True State", vmin=vmin, vmax=vmax, cmap=my_cmap)
fig.tight_layout()
plt.savefig("figures/true_state.png")
fig = plt.figure(figsize=default_figsize)
nb.plot_pts(targets, misfit.d, mytitle="Observations", vmin=vmin, vmax=vmax, cmap=my_cmap)
fig.tight_layout()
plt.savefig("figures/observations.png")

if strategy == Strategy.RMAP or strategy == Strategy.RMA_RMAP or strategy == Strategy.ENKF_U1:
    mean_parameter_solution = 0 
    for n in range(n_random_vectors):
        print(f'\nSolving {n + 1} / {n_random_vectors}')
        # independent realizations of noisy data
        misfit.d = observations.copy()
        parRandom.normal_perturb(noise_std_dev, misfit.d)

        prior.mean.zero()
        sample = priorSample(prior)
        prior.mean.set_local(sample)

        instance_solution = solveInstance(pde, prior, misfit)
        mean_parameter_solution += instance_solution[PARAMETER].get_local()
    mean_parameter_solution /= n_random_vectors
    solution = instance_solution
    solution[PARAMETER].set_local(mean_parameter_solution)

    # solve forward problem with mean parameter to get updated state
    pde.solveFwd(solution[STATE], solution)
else:
    solution = solveInstance(pde, prior, misfit)

# some output stuff
relative_error = np.linalg.norm(solution[PARAMETER].get_local() - true_sol.get_local()) / np.linalg.norm(true_sol.get_local())
print("SUMMARY:\n----------------------------------")
print(f'\tstrategy: {strategy}')
print(f'\tn_random_vecs: {n_random_vectors}')
print(f'\trelative_error to true: {relative_error}')

# save numpy solution for analysis later
filename = "solutions/{}_{}_mesh_{}.npy".format(strategy, n_random_vectors, Vh[PARAMETER].dim())
np.save(filename, solution[PARAMETER].get_local())

# plot results and save to file
fig = plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[STATE], solution[STATE]), mytitle="State", cmap=my_cmap)
fig.tight_layout()
plt.savefig("figures/final_state_{}_{}.png".format(strategy, n_random_vectors))

fig = plt.figure(figsize=default_figsize)
nb.plot(dl.Function(Vh[PARAMETER], solution[PARAMETER]), mytitle="Parameter", cmap=my_cmap, vmin=param_vmin, vmax=param_vmax)
fig.tight_layout()
plt.savefig("figures/final_parameter_{}_{}.png".format(strategy, n_random_vectors))
