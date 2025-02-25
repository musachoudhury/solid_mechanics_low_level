# %%
from sympy import symbols, Matrix, diff
import numpy as np
import basix

x_1, x_2, x_3, phi_1, E, nu, u_1 = symbols("x_1 x_2 x_3 phi_1 E nu u_1")

phi_1 = (1 - x_1) * (1 - x_2) * (1 - x_3)
phi_2 = x_1 * (1 - x_2) * (1 - x_3)
phi_3 = (1 - x_1) * (x_2) * (1 - x_3)
phi_4 = x_1 * (x_2) * (1 - x_3)
phi_5 = (1 - x_1) * (1 - x_2) * (x_3)
phi_6 = x_1 * (1 - x_2) * (x_3)
phi_7 = (1 - x_1) * (x_2) * (x_3)
phi_8 = x_1 * (x_2) * (x_3)

#phi = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8]
#

# %%
gradient_phi_1 = Matrix([])
for var in [x_1, x_2, x_3]:
    derivative = diff(phi_1, var)
    gradient_phi_1 = gradient_phi_1.row_join(Matrix([[derivative]]))

gradient_phi_1

gradient_phi_1_full = Matrix([gradient_phi_1 for i in range(3)])
gradient_phi_1_full

# %%
B = Matrix.zeros(6, 3)

B[0, 0] = diff(phi_1, x_1)
B[1, 1] = diff(phi_1, x_2)
B[2, 2] = diff(phi_1, x_3)
B[3, 1] = diff(phi_1, x_3)
B[3, 2] = diff(phi_1, x_2)
B[4, 0] = diff(phi_1, x_3)
B[4, 2] = diff(phi_1, x_1)
B[5, 0] = diff(phi_1, x_2)
B[5, 1] = diff(phi_1, x_1)
B
# %%
gradient_phi_1_full_transpose = gradient_phi_1_full.T
# %%
epsilon = 0.5 * (gradient_phi_1_full + gradient_phi_1_full_transpose)
epsilon
# %%
def voigt(u):
    return Matrix(
[u[0, 0],
u[1, 1],
u[2, 2],
2 * u[1, 2],
2 * u[0, 2],
2 * u[0, 1],])


epsilon_voigt = voigt(epsilon)


# %%
def tangent():
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    tangentArr = Matrix.zeros(6, 6)
    for i in range(0, 3):
        for j in range(0, 3):
            tangentArr[i, j] = lmbda
        tangentArr[i, i] = lmbda + 2.0 * mu

    for i in range(3, 6):
        tangentArr[i, i] = mu

    return tangentArr
tangent()

result = -B.T * tangent() * B
# np.set_printoptions(precision=3)
result2 = result.subs({x_1: 0.5, x_2: 0.5, x_3: 0.5, E: 200000, nu: 0.3})
result2
# %%
#tang = tangent()
vector1 = tangent() * epsilon_voigt
vector1
# %%
# #vector1 = tang.dot(epsilon_voigt)
# vector1 = epsilon_voigt.dot(tang)

dot_product = -vector1.dot(epsilon_voigt.subs({x_1: 0.5, x_2: 0.5, x_3: 0.5}))
dot_product
result = dot_product.subs({x_1: 0.5, x_2: 0.5, x_3: 0.5, E: 200000, nu: 0.3})
result
# %%
# quadrature_points, weights = basix.make_quadrature(basix.CellType.hexahedron, 1)

# %%
