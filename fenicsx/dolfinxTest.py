# %%
import pyvista
import numpy as np
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem, io, default_scalar_type, plot
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc
from utils import eps, interpolate_quadrature, set_meshtags, symmetry_bc, constitutive
from umat_interface import call_umat

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [1, 1, 1],
    mesh.CellType.hexahedron,
    ghost_mode=mesh.GhostMode.shared_facet,
)

gdim = domain.topology.dim
fdim = gdim - 1
degree = 1
quadrature_degree = 1
quadrature_rule = "default"
shape = (gdim,)


V = fem.functionspace(domain, ("P", degree, shape))


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dx = ufl.Measure("dx", domain=domain)

L = fem.form(ufl.inner(eps(u), eps(v))*dx)

# b = fem.petsc.create_vector(L)

# assemble_vector(b, L)