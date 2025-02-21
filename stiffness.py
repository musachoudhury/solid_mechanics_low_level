# %%
from sympy import symbols, Matrix, diff
import numpy as np
import basix


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


x_1, x_2, x_3, phi_1, E, nu, u_1 = symbols("x_1 x_2 x_3 phi_1 E nu u_1")

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
B = Matrix.zeros(6, 3)
A = Matrix.zeros(24, 24)


def calc_B(phi, x1, x2, x3):
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


# %%
for i in range(0, 8):
    for j in range(0, 8):
        # B.T * D * B
        result = (
            -calc_B(phi[i], x_1, x_2, x_3).T * tangent() * calc_B(phi[j], x_1, x_2, x_3)
        )

        # Quadrature integration
        result2 = result.subs({x_1: 0.5, x_2: 0.5, x_3: 0.5, E: 200000, nu: 0.3})
        A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = result2
A

# %%
#1,3,5,7

#Apply bc
# 1
i=0
A.col_op(3*i, lambda v, j: 0)
A.col_op(3*i+1, lambda v, j: 0)
A.col_op(3*i+2, lambda v, j: 0)

A.row_op(3*i, lambda v, j: 0)
A.row_op(3*i+1, lambda v, j: 0)
A.row_op(3*i+2, lambda v, j: 0)

# 2
i=1 
#A.col_op(3*i, lambda v, j: 0)
A.col_op(3*i+1, lambda v, j: 0)
A.col_op(3*i+2, lambda v, j: 0)

A.row_op(3*i+1, lambda v, j: 0)
A.row_op(3*i+2, lambda v, j: 0)

# 3
i=2
A.col_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
A.col_op(3*i+2, lambda v, j: 0)

A.row_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
A.row_op(3*i+2, lambda v, j: 0)

# 4
i=3
#A.col_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
A.col_op(3*i+2, lambda v, j: 0)

#A.col_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
A.row_op(3*i+2, lambda v, j: 0)


# 5
i=4
A.col_op(3*i, lambda v, j: 0)
A.col_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)

A.row_op(3*i, lambda v, j: 0)
A.row_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)


# 6
i=5 
#A.col_op(3*i, lambda v, j: 0)
A.col_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)

#A.col_op(3*i, lambda v, j: 0)
A.row_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)


# 7
i=6
A.col_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)

A.row_op(3*i, lambda v, j: 0)
#A.col_op(3*i+1, lambda v, j: 0)
#A.col_op(3*i+2, lambda v, j: 0)


A



# %%
