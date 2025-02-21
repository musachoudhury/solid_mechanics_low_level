import numpy as np
import ufl
from dolfinx import mesh, fem, default_scalar_type


def voigt(u):
    return ufl.as_vector(
        [u[0, 0], u[1, 1], u[2, 2], 2 * u[1, 2], 2 * u[0, 2], 2 * u[0, 1]]
    )


def eps(u):
    return voigt(ufl.sym(ufl.grad(u)))


def interpolate_quadrature(domain, expr, function):
    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=default_scalar_type)
    expr_eval = expr.eval(domain, cells)
    function.x.array[:] = expr_eval.flatten()[:]


def set_meshtags(domain, fdim, bc, tag):
    facets = mesh.locate_entities_boundary(domain, fdim, bc)
    marked_facets = np.hstack([facets])
    marked_values = np.hstack([np.full_like(facets, tag)])
    sorted_facets = np.argsort(marked_facets)
    return mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )


def symmetry_bc(V):
    sym_bc = []
    domain = V.mesh
    fdim = domain.topology.dim - 1
    for i in range(0, 3):
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, marker=lambda x: np.isclose(x[i], 0.0)
        )
        boundary_dofs = fem.locate_dofs_topological(V.sub(i), fdim, boundary_facets)
        bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V.sub(i))
        sym_bc.append(bc)
    return sym_bc


def constitutive(stressArr, tangentArr, dstrainArr):
    E = 200000.0
    nu = 0.3
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    for i in range(0, 36):
        tangentArr[i] = 0.0

    for i in range(0, 3):
        for j in range(0, 3):
            tangentArr[i * 6 + j] = lmbda
        tangentArr[i * 6 + i] = lmbda + 2.0 * mu

    for i in range(3, 6):
        tangentArr[i * 6 + i] = mu

    for i in range(0, 6):
        for j in range(0, 6):
            stressArr[i] = stressArr[i] + tangentArr[i * 6 + j] * dstrainArr[j]