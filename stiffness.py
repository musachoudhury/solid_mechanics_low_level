# %%
from sympy import symbols, Matrix, diff
import numpy as np
from petsc4py import PETSc

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


# %%
stressIn = Matrix.zeros(6, 1)
dstrain = Matrix.zeros(6, 1)
tangent = constitutive(stressIn, dstrain)

for i in range(0, 8):
    for j in range(0, 8):
        # B.T * D * B
        result = (
            -calc_B(phi[i], x_1, x_2, x_3).T * tangent * calc_B(phi[j], x_1, x_2, x_3)
        )

        # Quadrature integration
        result2 = result.evalf(subs = {x_1: 0.5, x_2: 0.5, x_3: 0.5, E: 200000.0, nu: 0.3}, n=15)
        A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = result2
#A

# %%

links = {
    0: [0, 1, 2],
    1: [1, 2],
    2: [0, 2],
    3: [2],
    4: [0, 1],
    5: [1],
    6: [0]
}

#Apply BC
for key in links:
    print(key)
    for i in links[key]:
        A.col_op(3*key+i, lambda v, j: 0.0)
        A.row_op(3*key+i, lambda v, j: 0.0)
        A[3*key+i, 3*key+i] = 1.0
A

#%%
body_force = Matrix([-bf, 0.0, 0.0])

N = Matrix.zeros(3, 24)

for index, value in enumerate(phi):
    for i in range(3):
        N[i, index*3+i] = value

#inner(b, v)*dx
b_vec = N.T*body_force

# %% Stress inner(sigma, eps(v))*dx
#b_stress = N.T*stress
# %%
for key in links:
    print(key)
    for i in links[key]:
        b_vec.row_op(3*key+i, lambda v, j: 0.0)
b_eval = b_vec.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5, bf: 1000.0}, n=15)
b_eval
# %%
#%%
A_np = np.array(A).astype(np.float64)
b_np = np.array(b_eval).astype(np.float64)

res = np.linalg.solve(A_np, b_np)
res
# %%

# %%
# Initialize PETSc objects
# A_petsc = PETSc.Mat().create()
# A_petsc.setSizes(A_np.shape)
# A_petsc.setType(PETSc.Mat.Type.DENSE)  # Set matrix type to dense
# A_petsc.setUp()  # Finalize setup before inserting values


# b_petsc = PETSc.Vec().createSeq(A_np.shape[0])
# x_petsc = PETSc.Vec().createSeq(A_np.shape[0])

# for i in range(A_np.shape[0]):
#     for j in range(A_np.shape[1]):
#         A_petsc.setValue(i, j, A_np[i, j])

# # Assemble the matrix
# A_petsc.assemble()
# #A_petsc.view()


# # Set right-hand side vector values
# for i in range(b_np.shape[0]):
#     b_petsc.setValue(i, b_np[i, 0])

# # Assemble the vector
# b_petsc.assemble()

# # Solve Ax = b using PETSc linear solver
# ksp = PETSc.KSP().create()
# ksp.setOperators(A_petsc)
# ksp.setType(PETSc.KSP.Type.PREONLY)  # Direct solve
# pc = ksp.getPC()
# pc.setType(PETSc.PC.Type.LU)  # Use LU factorization

# # Solve for x
# ksp.solve(b_petsc, x_petsc)

# # Retrieve the solution
# x_np = x_petsc.getArray()

# print("Solution x:", x_np)

# # %%
# A_petsc.convert("dense")
# array = A_petsc.getDenseArray()  # Get matrix as a NumPy array
# # # Save to a text file
# np.savetxt("matrix.txt", array, fmt="%.6f")  # Save with 6 decimal places
# %%

# Create a PETSc vector

# Save to text file
# viewer = PETSc.Viewer().createASCII("vector_output.txt")
# b_petsc.view(viewer)

res2d = res.reshape(3, 8)
res2d
# for i in range(0, 8):
#     for j in range(0, 8):
#         calc_B(phi[i], x_1, x_2, x_3)*
N_np = np.array(N.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5}, n=15)).astype(np.float64)
for i in range(0, 8):
    # u at QP
    u_qp = np.matmul(N_np,res)

    # strain = B *u_
    
    B = calc_B(phi[i], x_1, x_2, x_3)
    B_np = np.array(B.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5}, n=15)).astype(np.float64)

strain = np.matmul(B_np, res2d)
# %%
