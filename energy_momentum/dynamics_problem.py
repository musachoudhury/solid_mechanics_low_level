# %%

import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, io, nls
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from dolfinx.mesh import create_box, CellType, locate_entities_boundary
from ufl import (
    as_matrix,
    dot,
    cos,
    sin,
    SpatialCoordinate,
    Identity,
    grad,
    ln,
    tr,
    det,
    variable,
    derivative,
    TestFunction,
    TrialFunction,
)
import definitions as defs
from math import sin, pi

L = 3.0
N = 4
a = 0.5
b = 0.5
c = 6.0
mesh = create_box(
    MPI.COMM_WORLD,
    [[-a / 2, -b / 2, 0.0], [a / 2, b / 2, c]],
    [2, 2, 8],
    CellType.hexahedron,
)


def pressure_surface(x):
    # height = np.less(x[2], c/12)
    height = np.greater(x[2], c - c / 12)
    return height


dim = mesh.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(mesh, ("P", degree, shape))

v = ufl.TestFunction(V)
# Identity tensor
Id = Identity(dim)

# Shear modulus
Eym = 200e3
nu = 0.3
mu = fem.Constant(mesh, Eym / 2 / (1 + nu))
lmbda = fem.Constant(mesh, Eym * nu / (1 - 2 * nu) / (1 + nu))

# Step 1

d_n_1 = fem.Function(V)
d_n = fem.Function(V)

V_n = fem.Function(V)
V_n_1 = fem.Function(V)


def Dphi(u):
    return variable(Id + grad(u))


def Dphi_alpha(alpha):
    return alpha * Dphi(d_n_1) + (1 - alpha) * Dphi(d_n)


# Dphi_n_alpha = alpha*Dphi(d_n_1)+(1-alpha)*Dphi(d_n)

F_test = variable(Id + grad(d_n_1))
C_test = F_test.T * F_test
E_test = variable(0.5 * (C_test - Id))

Dphi_n = Dphi_alpha(0.0)  # alpha = 0
Dphi_n_half = Dphi_alpha(0.5)  # alpha = 0.5
Dphi_n_1 = Dphi_alpha(1.0)  # alpha = 1

# Right Cauchy Green tensors


def C(F):
    return F.T * F


C_n = C(Dphi_n)
C_n_1 = C(Dphi_n_1)

# Green Langrange strain tensor


def E(C):
    return variable(0.5 * (C - Id))


# C_mat =


def e_hat(C):
    return 0.5 * lmbda * ufl.tr(E(C)) ** 2 + mu * ufl.inner(E(C), E(C))


# quadratic stored energy function
def grad_e(C):
    C_variable = variable(C)
    return ufl.diff(e_hat(C_variable), (C_variable))


def grad2_e(C):
    C_variable = variable(C)
    return ufl.diff(grad_e(C_variable), (C_variable))


# g'
# dg = e_hat(C_n_1)-e_hat(C_n)-0.5*ufl.inner(grad_e(C_n_beta_0)+grad_e(C_n_1_beta_0), C_n_1-C_n)

# beta_0 = 0.5 for Saint Venant Kirchoff
beta_0 = fem.Constant(mesh, 0.5)
dt = fem.Constant(mesh, 0.0)
rho = fem.Constant(mesh, 0.0)
b = fem.Constant(mesh, [0.0, 0.0, 0.0])
dT = fem.Constant(mesh, (0.0))

C_n_beta_0 = beta_0 * C_n_1 - (1 - beta_0) * C_n
C_n_1_beta_0 = (1 - beta_0) * C_n_1 - (1 - (1 - beta_0)) * C_n
# PK2 stress = 2*d_psi/d_C
# S = grad_e(C_n_1)  # ufl.diff(e_hat(C_n_1), (C_n_1))

S = grad_e(C_n_beta_0) + grad_e(C_n_1_beta_0)
C_mat = 2 * (beta_0 * grad2_e(C_n_beta_0) + (1 - beta_0) * grad2_e(C_n_1_beta_0))


def set_meshtags(domain, fdim, bc, tag):
    facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, bc)
    marked_facets = np.hstack([facets])
    marked_values = np.hstack([np.full_like(facets, tag)])
    sorted_facets = np.argsort(marked_facets)
    return dolfinx.mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )


pressure_surface_tags = set_meshtags(mesh, dim - 1, pressure_surface, 1)


dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=pressure_surface_tags)

Fint_n_half = ufl.inner(Dphi_n_half * S, ufl.grad(v)) * dx

traction = fem.Constant(mesh, [0.0, 0.0, 0.0])
traction.value = np.array(
    [0.0, 0.0, 0.0]
)  # defs.mechanics.forces([100, 0.0, 0.0], [1.0, 1.0])

Fext_n_half = ufl.inner(b, v) * dx + ufl.dot(traction * dT, v) * ds(1)


residual = (
    rho * 2 / dt / dt * ufl.inner(d_n_1 - d_n - dt * V_n, v) * dx
    + Fint_n_half
    - Fext_n_half
)
# S.ufl_shape
du = ufl.TrialFunction(V)

deltaD = fem.Function(V)

i, j, k, l = ufl.indices(4)

# %%
K_ab = (
    ufl.inner(grad(du), S * grad(v)) * dx
    + ufl.inner(
        Dphi_n_half.T * grad(v),
        ufl.as_tensor(
            (C_mat[i, j, k, l] * (Dphi_alpha(beta_0).T * grad(du))[k, l]), (i, j)
        ),
    )
    * dx
)

jacobian = -(2 / dt / dt * rho * ufl.inner(v, du) * dx + K_ab)

v_expr = fem.Expression((d_n_1 - d_n) / dt, V.element.interpolation_points)

###################################################


def left(x):
    return np.isclose(x[2], 0.0)


clamped_dofs = fem.locate_dofs_geometrical(V, left)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]
# bcs = []

a_form = fem.form(jacobian)
L_form = fem.form(residual)

check = ufl.inner(S, ufl.as_tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) * dx
check_form = fem.form(check)

problem = fem.petsc.LinearProblem(
    a_form,
    L_form,
    u=deltaD,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

Nsteps = 10
Nsave = 100
times = np.linspace(0, 0.1, Nsteps + 1)
save_freq = Nsteps // Nsave

iter = 1

relative_residual = 1
residual = 1
rtol = 1e-4
atol = 1e-4
max_it = 25

t = 0.0

vtk = io.VTKFile(mesh.comm, "results/elastodynamics.pvd", "w")

# %%

for i, dti in enumerate(np.diff(times)):
    if i % 1 == 0:
        vtk.write_function(d_n_1, t)

    dt.value = dti
    t += dti

    # print(dti)

    if t <= 0.2:
        dT.value = 0.0

    # dT.value =  sin(4*pi*t)
    else:
        dT.value *= 0.0

    while ((relative_residual > rtol and residual > atol) or False) and iter <= max_it:
        problem.solve()

        deltaD.x.scatter_forward()  # updates ghost values for parallel computations

        d_n_1.x.petsc_vec.axpy(1, deltaD.x.petsc_vec)

        d_n_1.x.scatter_forward()

        V_n_1.interpolate(v_expr)

        # SPS.interpolate(SPS_expr)

        if iter == 1:
            residual0 = deltaD.x.petsc_vec.norm()
        residual = problem.b.norm()
        du_norm = deltaD.x.petsc_vec.norm()
        relative_residual = residual / residual0

        iter += 1
        # print(f"check = {fem.assemble_scalar(check_form)}")

    if (relative_residual < rtol or residual < atol) and (iter <= max_it):
        print(
            f"(converged) Newton iteration {iter - 1} residual: {residual} relative_residual: {relative_residual} du norm: {du_norm}"
        )
    else:
        print(
            f"(Newton solver failed to converge) Newton iteration {iter - 1} residual: {residual} relative_residual: {relative_residual} du norm: {du_norm}"
        )

    # print(f"check = {fem.assemble_scalar(check_form)}")

    d_n_1.x.petsc_vec.copy(d_n.x.petsc_vec)
    V_n_1.x.petsc_vec.copy(V_n.x.petsc_vec)

    iter = 1
    relative_residual = 1
    residual = 1
