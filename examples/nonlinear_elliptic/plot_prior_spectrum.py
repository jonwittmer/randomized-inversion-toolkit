import sys
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
from hippylib import *
from randomized_priors import BiLaplacianPriorRand, fenics_operator_to_numpy

nx = 24
ny = 39
alpha = np.pi / 4.0
gamma = 1.0
delta = 8.0
theta0 = 2.0
theta1 = 0.5

mesh = dl.UnitSquareMesh(nx, ny)
V = dl.FunctionSpace(mesh, 'Lagrange', 1)
print(f'DoFs: {V.dim()}')

anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree=1)
anis_diff.set(theta0, theta1, alpha)

n_samples = [100, 500, 1000, 10000]

# true prior
true_prior = BiLaplacianPriorRand(V, gamma, delta, anis_diff, robin_bc=True)
matrix_representation = true_prior.R.R_half @ true_prior.R.R_half.T
_, true_spectrum, _ = np.linalg.svd(matrix_representation)

# compute spectrum of randomized prior
rand_spectrum = []
for n in n_samples:
    random_vectors = 1. / (n)**(0.5) * np.random.normal(0, 1.0, (n, V.dim()))
    prior = BiLaplacianPriorRand(V, gamma, delta, anis_diff, robin_bc=True, random_vectors=random_vectors)
    matrix_representation = fenics_operator_to_numpy(prior.implicit_prior.M.mpi_comm(), prior.R.mult, prior.prior_dim)
    _, s, _ = np.linalg.svd(matrix_representation)
    rand_spectrum.append(s)

plt.rcParams.update({ 'text.usetex': True,
                      'font.size': 14,
                      })
# set the line styles here to match plot already in paper
line_colors = ['blue', 'red', 'green', 'orange']
line_styles = ['o-', 'x-', 'D-', '^-']
fig, ax = plt.subplots()
ax.plot(range(V.dim()), true_spectrum, 'k--', label=r"true $\mathcal{C}^{-1}$")
for n, s, style, color in zip(n_samples, rand_spectrum, line_styles, line_colors):
    ax.plot(range(V.dim()), s, style, color=color, label=f"N = {n}", markevery=100)
ax.set_ylim(None, 5e5)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(5,5))
ax.legend()
ax.set_title('BiLaplacian prior', weight='bold')
plt.savefig('bilaplacian_prior_spectrum', dpi=300, format='pdf')

# identity prior
# true prior
true_spectrum = np.ones((V.dim(),))

# compute spectrum of randomized prior
rand_spectrum = []
for n in n_samples:
    random_vectors = 1. / (n)**(0.5) * np.random.normal(0, 1.0, (n, V.dim()))
    _, s, _ = np.linalg.svd(random_vectors.T @ random_vectors)
    rand_spectrum.append(s)

plt.rcParams.update({ 'text.usetex': True,
                      'font.size': 14,
                      })
# set the line styles here to match plot already in paper
line_colors = ['blue', 'red', 'green', 'orange']
line_styles = ['o-', 'x-', 'D-', '^-']
fig, ax = plt.subplots()
ax.plot(range(V.dim()), true_spectrum, 'k--', label=r"true $\mathcal{C}^{-1}$")
for n, s, style, color in zip(n_samples, rand_spectrum, line_styles, line_colors):
    ax.plot(range(V.dim()), s, style, color=color, label=f"N = {n}", markevery=100)
ax.set_ylim(None, 5)
ax.legend()
ax.set_title('Identity prior', weight='bold')
plt.savefig('identity_prior_spectrum', dpi=300, format='pdf')
plt.show()


