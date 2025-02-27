import ufl
from basix.ufl import element, quadrature_element
from basix import make_quadrature, CellType

# Function spaces
element_type = "hexahedron"
e = element("Lagrange", element_type, 1, shape=(3,))
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
C = ufl.Coefficient(Q_36)
D = ufl.Coefficient(Q_36)

b = ufl.Constant(mesh, shape=(3,))
T = ufl.Constant(mesh, shape=(3,))
dT = ufl.Constant(mesh)
n = ufl.FacetNormal(mesh)

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
            u[
                1,
                1,
            ],
            u[2, 2],
            2 * u[1, 2],
            2 * u[0, 2],
            2 * u[0, 1],
        ]
    )


# def eps(u):
#     return voigt(ufl.sym(ufl.grad(u)))

#print(ufl.Dx(u[0], 1))

def B(u):
    return ufl.as_matrix([[ufl.Dx(u[0], 0), 0, 0],
                          [0, ufl.Dx(u[1], 1), 0],
                          [0, 0, ufl.Dx(u[2], 2)],
                          [0, ufl.Dx(u[1], 2), ufl.Dx(u[2], 1)],
                          [ufl.Dx(u[0], 2), 0, ufl.Dx(u[2], 0)],
                          [ufl.Dx(u[0], 1), ufl.Dx(u[1], 0), 0]])

B(u)
    
def unit():
    return ufl.as_matrix([[1], [1], [1]])

#print(len(unit()))
#B(u)
#a = B(u).T*D
# a = 0   
# for i in range(3):
#     for j in range(3):
#         a += (((B(u).T)*D)*B(v))[i, j]

# a_form = a*dx
K = ((((B(u)*unit()).T)*D)*(B(v)*unit()))
print(K.ufl_shape)
#print(((((B(u)*unit()).T)*D)*(B(v)*unit())))
a_form = ((((B(u)*unit()).T)*D)*(B(v)*unit()))[0, 0]*dx

# a_form = ufl.inner(((((u*B(u))).T)*D), (B(v)*v))*dx
# a_form = -ufl.inner(ufl.dot(C, eps(du)), eps(v)) * dx
# l_form = (
#     ufl.inner(stress, eps(v)) * dx - ufl.inner(T * dT, v) * ds(1) - ufl.inner(b, v) * dx
# )

# # Projection

# Se = element("DG", element_type, 1, shape=(6,))
# S = ufl.FunctionSpace(mesh, Se)

# u_p = ufl.TrialFunction(S)  # Function to be projected onto
# w = ufl.TestFunction(S)
# u_original = ufl.Coefficient(Q_6)  # Function to be projected from
# kappa = ufl.Constant(S)

# a_p = ufl.inner(u_p, w) * dx
# L_p = ufl.inner(u_original, w) * dx

# # Form to calculate the reaction forces
# v_reac = ufl.Coefficient(V)
# reaction = ufl.action(
#     ufl.action(ufl.inner(stress, eps(v)) * dx, v)
#     - (-ufl.inner(T * dT, v) * ds(1) - ufl.inner(b, v) * dx),
#     v_reac,
# )

# forms = [a_form, l_form, a_p, L_p, reaction]

# elements = [e, Qe_36, Qe_6]

# x = ufl.SpatialCoordinate(mesh)

# x_expr = x
# Q_expr_6 = eps(u)
# expressions = [(Q_expr_6, pts), (x_expr, pts)]
