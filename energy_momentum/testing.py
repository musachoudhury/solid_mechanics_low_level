import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem
import dolfinx.fem.petsc
import dolfinx.nls.petsc

from ufl import (
    Identity,
    grad,
    variable,
)


L = 3.0
N = 4
a = 0.5
b = 0.5
c = 6.0
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (L, 1.0, 1.0)], [50, 12, 12]
)

dim = mesh.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(mesh, ("P", degree, shape))

# Identity tensor
Id = ufl.variable(Identity(dim))

# Shear modulus
Eym = 200e3
nu = 0.3
mu = fem.Constant(mesh, Eym / 2 / (1 + nu))
lmbda = fem.Constant(mesh, Eym * nu / (1 - 2 * nu) / (1 + nu))

# Step 1

d_n_1 = fem.Function(V)


# def Dphi(u):
#     return variable(Id + grad(u))

F = Id + grad(d_n_1)


def E(C):
    return 0.5 * (C - Id)


# Right Cauchy Green tensors


def C(F):
    return F.T * F


C_F = variable(C(F))
# e_hat = 0.5 * lmbda * ufl.tr(E(C_F)) ** 2 + mu * ufl.inner(E(C_F), E(C_F))


def e_hat(C):
    return 0.5 * lmbda * ufl.tr(E(C)) ** 2 + mu * ufl.inner(E(C), E(C))


S = ufl.diff(e_hat(C_F), C_F)

# C = variable(F.T * F)


# e_hat = 0.5 * lmbda * ufl.tr(E(C)) ** 2 + mu * ufl.inner(E(C), E(C))
# S = ufl.diff(e_hat, C)

metadata = {"quadrature_degree": 3}
dx = ufl.Measure("dx", metadata=metadata)

d_n_1.interpolate(lambda x: np.vstack((x[0], x[1], x[2])))

check = ufl.inner(S, ufl.as_tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) * dx
check_form = fem.form(check)
print(f"check = {fem.assemble_scalar(check_form)}")
