import ufl
from basix.ufl import element, quadrature_element
from basix import make_quadrature, CellType

# Function spaces
element_type = "hexahedron"
e = element("Lagrange", element_type, 2, shape=(3,))
mesh = ufl.Mesh(e)
V = ufl.FunctionSpace(mesh, e)

# Trial and test functions
du = ufl.TrialFunction(V)  # Incremental displacement
v = ufl.TestFunction(V)  # Test function

# Quadrature functions
quadrature_degree = 2
quadrature_rule = "default"

Qe_36 = quadrature_element(
    element_type, (6, 6), degree=quadrature_degree, scheme="default"
)

Qe_6 = quadrature_element(
    element_type, (6,), degree=quadrature_degree, scheme="default"
)

Q_36 = ufl.FunctionSpace(mesh, Qe_36)
Q_6 = ufl.FunctionSpace(mesh, Qe_6)

u = ufl.Coefficient(V)

stress = ufl.Coefficient(Q_6)
tangent = ufl.Coefficient(Q_36)

pts, wts = make_quadrature(CellType.hexahedron, quadrature_degree)

metadata = {
    "quadrature_degree": quadrature_degree,
    "quadrature_scheme": quadrature_rule,
}

ds = ufl.Measure("ds", metadata=metadata, domain=mesh)
dx = ufl.Measure("dx", domain=mesh, metadata=metadata)


def voigt(u):
    return ufl.as_vector(
        [
            u[0, 0],
            u[1, 1],
            u[2, 2],
            2 * u[1, 2],
            2 * u[0, 2],
            2 * u[0, 1],
        ]
    )


def eps(u):
    return voigt(ufl.sym(ufl.grad(u)))


a = -ufl.inner(ufl.dot(tangent, eps(du)), eps(v)) * dx
L = ufl.inner(stress, eps(v)) * dx
