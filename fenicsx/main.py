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

Qe_36 = basix.ufl.quadrature_element(
    domain.basix_cell(),
    value_shape=(6, 6),
    scheme=quadrature_rule,
    degree=quadrature_degree,
)
Qe_6 = basix.ufl.quadrature_element(
    domain.basix_cell(),
    value_shape=(6,),
    scheme=quadrature_rule,
    degree=quadrature_degree,
)
Qe_stateVars = basix.ufl.quadrature_element(
    domain.basix_cell(),
    value_shape=(1,),
    scheme=quadrature_rule,
    degree=quadrature_degree,
)

Q_36 = fem.functionspace(domain, Qe_36)
Q_6 = fem.functionspace(domain, Qe_6)
Q_stateVars = fem.functionspace(domain, Qe_stateVars)
# %%

zeroArray = [0.0, 0.0, 0.0]

T = fem.Constant(domain, np.array([-0.0, 0.0, 0.0]))
n = ufl.FacetNormal(domain)
dT = fem.Constant(domain, default_scalar_type(0.0))
bf = fem.Constant(domain, np.array([0.0, 0.0, 0.0]))

stress = fem.Function(Q_6)
stress_old = fem.Function(Q_6)
tangent = fem.Function(Q_36)

statev = fem.Function(Q_stateVars)
statev_old = fem.Function(Q_stateVars)
strain = fem.Function(Q_6)
dstrain = fem.Function(Q_6)

utf = ufl.TrialFunction(V)
# du = ufl.TrialFunction(V)
du = fem.Function(V)
u = fem.Function(V)
Du = fem.Function(V)

v = ufl.TestFunction(V)

metadata = {
    "quadrature_degree": quadrature_degree,
    "quadrature_scheme": quadrature_rule,
}

pressure_surface_tags = set_meshtags(domain, fdim, lambda x: np.isclose(x[0], 1.0), 1)

bcs = symmetry_bc(V)

ds = ufl.Measure("ds", domain=domain, subdomain_data=pressure_surface_tags)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)


a_form = -ufl.inner(ufl.dot(tangent, eps(utf)), eps(v)) * dx
L_form = (
    ufl.inner(stress, eps(v)) * dx
    - ufl.inner(T * dT, v) * ds(1)
    - ufl.inner(bf, v) * dx
)

v_reac = fem.Function(V)
reaction = fem.form(
    ufl.action(
        ufl.action(ufl.inner(stress, eps(v)) * dx, v)
        - (-ufl.inner(T * dT, v) * ds(1) - ufl.inner(bf, v) * dx),
        v_reac,
    )
)

check = fem.form(ufl.inner(stress, eps(v)) * dx)
c = fem.petsc.create_vector(check)

a = fem.form(a_form)
L = fem.form(L_form)

basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, quadrature_degree)

print(weights)
print(quadrature_points)
# exit()

map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.float64)

strain_expr = fem.Expression(eps(u), quadrature_points)
dstrain_expr = fem.Expression(eps(Du), quadrature_points)

numQPointsLocal = num_cells * quadrature_points.shape[0]

A = fem.petsc.create_matrix(a)
b = fem.petsc.create_vector(L)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)

solver.setType("preonly")
solver.getPC().setType("lu")
# solver.setTolerances(rtol=1e-6)
# solver.setTolerances(atol=1e-6)

iter = 1
max_iterations = 25

interpolate_quadrature(domain, strain_expr, strain)
interpolate_quadrature(domain, dstrain_expr, dstrain)
# %%

for i in range(0, numQPointsLocal):
    # call_umat(
    #     stress.x.array[6 * i : 6 * (i + 1)],
    #     statev.x.array[1 * i : 1 * (i + 1)],
    #     tangent.x.array[36 * i : 36 * (i + 1)],
    #     dstrain.x.array[6 * i : 6 * (i + 1)],
    # )
    constitutive(
        stress.x.array[6 * i : 6 * (i + 1)],
        tangent.x.array[36 * i : 36 * (i + 1)],
        dstrain.x.array[6 * i : 6 * (i + 1)],
    )

print(tangent.x.array)
np.set_printoptions(precision=3)

relative_residual = 1
residual = 1
rtol = 1e-4
atol = 1e-4
max_it = 25


A.zeroEntries()
fem.petsc.assemble_matrix(A, a, bcs=bcs)
A.assemble()

A.view()

#viewer = PETSc.Viewer().createASCII("matrix.txt", mode="w")
#viewer = PETSc.Viewer().createBinary("matrix.dat", mode="w")
#A.view(viewer)
# #print(A.getValue(2, 2))
A.convert("dense")
array = A.getDenseArray()  # Get matrix as a NumPy array

# # Save to a text file
np.savetxt("matrix.txt", array, fmt="%.6f")  # Save with 6 decimal places


exit()
# %%
dT.value = 0.0
for num in range(0, 22):
    dT.value += 0.05
    print(dT.value)
    Du.x.petsc_vec.zeroEntries()
    while ((relative_residual > rtol and residual > atol) or False) and iter <= max_it:
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, a, bcs=bcs)
        A.assemble()

        with b.localForm() as loc_b:
            loc_b.set(0)

        fem.petsc.assemble_vector(b, L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # Compute b - J(u_D-u_(i-1))
        fem.petsc.apply_lifting(b, [a], [bcs])
        # Set du|_bc = u_{i-1}-u_D
        fem.petsc.set_bc(b, bcs, u.x.petsc_vec, 1.0)

        # Solve linear problem
        solver.solve(b, du.x.petsc_vec)
        du.x.scatter_forward()

        # Update
        Du.x.petsc_vec.axpy(1, du.x.petsc_vec)

        Du.x.scatter_forward()

        interpolate_quadrature(domain, strain_expr, strain)
        interpolate_quadrature(domain, dstrain_expr, dstrain)

        stress_old.x.petsc_vec.copy(stress.x.petsc_vec)
        statev_old.x.petsc_vec.copy(statev.x.petsc_vec)
        for i in range(0, numQPointsLocal):
            call_umat(
                stress.x.array[6 * i : 6 * (i + 1)],
                statev.x.array[1 * i : 1 * (i + 1)],
                tangent.x.array[36 * i : 36 * (i + 1)],
                dstrain.x.array[6 * i : 6 * (i + 1)],
            )

        assemble_vector(b, L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.apply_lifting(b, [a], [bcs])
        fem.petsc.set_bc(b, bcs, u.x.petsc_vec, 1.0)

        if iter == 1:
            residual0 = du.x.petsc_vec.norm()
        residual = b.norm()
        du_norm = du.x.petsc_vec.norm()
        relative_residual = residual / residual0

        # print(f"(converged) Newton iteration {iter-1} residual: {residual} relative_residual: {relative_residual} du norm: {du_norm}")

        # print(relative_residual)
        iter += 1
    if (relative_residual < rtol or residual < atol) and (iter <= max_it):
        print(
            f"(converged) Newton iteration {iter - 1} residual: {residual} relative_residual: {relative_residual} du norm: {du_norm}"
        )
    else:
        print(
            f"(Newton solver failed to converge) Newton iteration {iter - 1} residual: {residual} relative_residual: {relative_residual} du norm: {du_norm}"
        )

    print(residual)
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, marker=lambda x: np.isclose(x[0], 1.0)
    )
    bdofsRx = fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
    bcRx = fem.dirichletbc(default_scalar_type(1.0), bdofsRx, V.sub(0))
    fem.petsc.set_bc(v_reac.x.petsc_vec, [bcRx], None, 1.0)
    # fem.set_bc(v_reac.x.array[:], [bcRx])
    print(f"Reaction force: {fem.assemble_scalar(reaction)}")
    iter = 1
    relative_residual = 1
    residual = 1
    stress.x.petsc_vec.copy(stress_old.x.petsc_vec)
    statev.x.petsc_vec.copy(statev_old.x.petsc_vec)
    u.x.petsc_vec.axpy(-1, Du.x.petsc_vec)

# Form to calculate the reaction forces

# %%
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, marker=lambda x: np.isclose(x[0], 1.0)
)
boundary_dofs = fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets)
bcRx = fem.dirichletbc(default_scalar_type(1.0), boundary_dofs, V.sub(0))
# fem.petsc.set_bc(v_reac.x.petsc_vec, [bcRx], None, 1.0)
fem.set_bc(v_reac.x.array[:], [bcRx])
print(f"Reaction force: {fem.assemble_scalar(reaction)}")
# %%

with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    u.name = "Deformation"
    xdmf.write_function(u)

# from dolfinx.io.utils import VTKFile

# A_file = VTXWriter(domain.comm, "deformation.bp", u, "BP4")

pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")
