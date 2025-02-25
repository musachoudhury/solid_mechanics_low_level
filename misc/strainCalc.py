# %%
from sympy import symbols, Matrix, diff
import numpy as np
from petsc4py import PETSc
import basix

quadrature_points, weights = basix.make_quadrature(basix.CellType.hexahedron, 2)


def constitutive(stressArr, dstrainArr):
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    tangentArr = Matrix.zeros(6, 6)
    for i in range(0, 3):
        for j in range(0, 3):
            tangentArr[i, j] = lmbda
        tangentArr[i, i] = lmbda + 2.0 * mu

    for i in range(3, 6):
        tangentArr[i, i] = mu

    # for i in range(0, 6):
    #     for j in range(0, 6):
    #         stressArr[i, 1] = stressArr[i, 1] + tangentArr[i, j] * dstrainArr[j, 1]

    return tangentArr


x_1, x_2, x_3, phi_1, E, nu, u_1, bf = symbols("x_1 x_2 x_3 phi_1 E nu u_1 bf")

phi_1 = (1 - x_1) * (1 - x_2) * (1 - x_3)  # (0,0,0) 1
phi_2 = x_1 * (1 - x_2) * (1 - x_3)  # (1,0,0) 2
phi_3 = (1 - x_1) * (x_2) * (1 - x_3)  # (0,1,0) 3
phi_4 = x_1 * (x_2) * (1 - x_3)  # (1,1,0) 4
phi_5 = (1 - x_1) * (1 - x_2) * (x_3)  # (0,0,1) 5
phi_6 = x_1 * (1 - x_2) * (x_3)  # (1,0,1) 6
phi_7 = (1 - x_1) * (x_2) * (x_3)  # (0,1,1) 7
phi_8 = x_1 * (x_2) * (x_3)  # (1,1,1) 8

phi = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8]
# %%
u = Matrix.zeros(3, 8)
B = Matrix.zeros(6, 3)
A = Matrix.zeros(24, 24)


def calc_B(phi, x_1, x_2, x_3):
    B[0, 0] = diff(phi, x_1)
    B[1, 1] = diff(phi, x_2)
    B[2, 2] = diff(phi, x_3)
    B[3, 1] = diff(phi, x_3)
    B[3, 2] = diff(phi, x_2)
    B[4, 0] = diff(phi, x_3)
    B[4, 2] = diff(phi, x_1)
    B[5, 0] = diff(phi, x_2)
    B[5, 1] = diff(phi, x_1)
    return B
