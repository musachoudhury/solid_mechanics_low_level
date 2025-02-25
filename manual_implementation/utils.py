# %%
from sympy import Matrix, diff

def constitutive(stressArr, dstrainArr, E, nu):
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    tangentArr = Matrix.zeros(6, 6)
    for i in range(0, 3):
        for j in range(0, 3):
            tangentArr[i, j] = lmbda
        tangentArr[i, i] = lmbda + 2.0 * mu

    for i in range(3, 6):
        tangentArr[i, i] = mu

    for i in range(0, 6):
        for j in range(0, 6):
            stressArr[i, 0] = stressArr[i, 0] + tangentArr[i, j] * dstrainArr[j, 0]
    stressArr
    return tangentArr, stressArr

def getBasisFunctions(x_1, x_2, x_3):
    phi_1 = (1 - x_1) * (1 - x_2) * (1 - x_3)  # (0,0,0) 1
    phi_2 = x_1 * (1 - x_2) * (1 - x_3)  # (1,0,0) 2
    phi_3 = (1 - x_1) * (x_2) * (1 - x_3)  # (0,1,0) 3
    phi_4 = x_1 * (x_2) * (1 - x_3)  # (1,1,0) 4
    phi_5 = (1 - x_1) * (1 - x_2) * (x_3)  # (0,0,1) 5
    phi_6 = x_1 * (1 - x_2) * (x_3)  # (1,0,1) 6
    phi_7 = (1 - x_1) * (x_2) * (x_3)  # (0,1,1) 7
    phi_8 = x_1 * (x_2) * (x_3)  # (1,1,1) 8

    phi = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8]

    return phi

def calc_B(phi, x_1, x_2, x_3):
    B = Matrix.zeros(6, 3)
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


def assemble_matrix(A, phi, x_1, x_2, x_3, E, nu, weights, quadrature_points, tangent):        
    for i in range(0, 8):
        for j in range(0, 8):
            result2 = Matrix.zeros(3, 3)
            # B.T * D * B
            result = (
                -calc_B(phi[i], x_1, x_2, x_3).T * tangent * calc_B(phi[j], x_1, x_2, x_3)
            )

            # Quadrature integration
            for index, w in enumerate(weights):
                result2 += w * result.evalf(
                    subs={
                        x_1: quadrature_points[index][0],
                        x_2: quadrature_points[index][1],
                        x_3: quadrature_points[index][2],
                        E: 200000.0,
                        nu: 0.3,
                    },
                )
            A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = result2

    return A

def assemble_vector(b, phi, x_1, x_2, x_3, body_force, weights, quadrature_points, stress):
    for i, value in enumerate(phi):
        N = Matrix.zeros(3, 3)
        B_T = calc_B(phi[i], x_1, x_2, x_3).T

        for j in range(3):
            N[j, j] = value

        b_val = N.T * body_force

        for index, w in enumerate(weights):
            f_int = B_T * stress[:, index]

            f = b_val + f_int

            b[3*i:3*i+3, 0] += w * f.evalf(
            subs={
                x_1: quadrature_points[index][0],
                x_2: quadrature_points[index][1],
                x_3: quadrature_points[index][2],
            })
        
    return b

def set_bc(A, b, bcs):
    for key in bcs:
        print(key)
        for i in bcs[key]:
            A.col_op(3 * key + i, lambda v, j: 0.0)
            A.row_op(3 * key + i, lambda v, j: 0.0)
            A[3 * key + i, 3 * key + i] = 1.0

    for key in bcs:
    #   print(key)
        for i in bcs[key]:
            b.row_op(3 * key + i, lambda v, j: 0.0)
            
    return A, b

def getStrainExpr(u, phi, x_1, x_2, x_3):
    B_global = Matrix.zeros(6, 24)
    for i in range(0, 8):
        B_global[:, i*3:i*3+3] = calc_B(phi[i], x_1, x_2, x_3)
    strain = B_global*u

    return strain