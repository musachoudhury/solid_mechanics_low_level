# %%
import os
os.chdir("/home/musa/Documents/main_projects/my_demos/solid_mechanics_low_level/manual_implementation") 

from sympy import symbols, Matrix, diff
from utils import constitutive, getBasisFunctions, calc_B, assemble_matrix
import basix


quadrature_points, weights = basix.make_quadrature(basix.CellType.hexahedron, 2)

x_1, x_2, x_3, phi_1, E, nu, u_1, bf = symbols("x_1 x_2 x_3 phi_1 E nu u_1 bf")

phi = getBasisFunctions(x_1, x_2, x_3)
# %%
u = Matrix.zeros(24, 1)
B = Matrix.zeros(6, 3)
A = Matrix.zeros(24, 24)

# %%
stressIn = Matrix.zeros(6, 1)
dstrain = Matrix.zeros(6, 1)
tangent, stressIn = constitutive(stressIn, dstrain, E, nu)

A = assemble_matrix(A, phi, x_1, x_2, x_3, E, nu, weights, quadrature_points, tangent)        

# %%

links = {0: [0, 1, 2], 1: [1, 2], 2: [0, 2], 3: [2], 4: [0, 1], 5: [1], 6: [0]}

# Apply BC
for key in links:
    print(key)
    for i in links[key]:
        A.col_op(3 * key + i, lambda v, j: 0.0)
        A.row_op(3 * key + i, lambda v, j: 0.0)
        A[3 * key + i, 3 * key + i] = 1.0
# A

# %%
body_force = Matrix([-bf, 0.0, 0.0])

N = Matrix.zeros(3, 24)

for index, value in enumerate(phi):
    for i in range(3):
        N[i, index * 3 + i] = value

# inner(b, v)*dx
b_vec = N.T * body_force

# %% Stress inner(sigma, eps(v))*dx
# b_stress = N.T*stress
# %%
for key in links:
    print(key)
    for i in links[key]:
        b_vec.row_op(3 * key + i, lambda v, j: 0.0)

b_eval = Matrix.zeros(24, 1)
for index, w in enumerate(weights):
    b_eval += w * b_vec.evalf(
        subs={
            x_1: quadrature_points[index][0],
            x_2: quadrature_points[index][1],
            x_3: quadrature_points[index][2],
            bf: 1000,
        }
    )


# for dof in range(0, 8):
#     result2 = Matrix.zeros(3, 1)
#     # ∫ B.T * σ dv
#     result = (
#         -calc_B(phi[dof], x_1, x_2, x_3).T )

# b_eval = b_vec.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5, bf: 1000.0}, n=20)
b_eval
# %%

du = A.LUsolve(b_eval)
du
u = u - du
u

# %%

stress = Matrix.zeros(6, len(weights))
dstrain = Matrix.zeros(6, len(weights))

# %%
B_global = Matrix.zeros(6, 24)
for i in range(0, 8):
    B_global[:, i*3:i*3+3] = calc_B(phi[i], x_1, x_2, x_3)
strain = B_global*du
strain.evalf(
            subs={
                x_1: 1.0,
                x_2: 0.0,
                x_3: 0.0,
            })

for index, w in enumerate(weights):
    
    tangent, stress[:, index] = constitutive(stress[:, index], strain.evalf(
            subs={
                x_1: quadrature_points[index][0],
                x_2: quadrature_points[index][1],
                x_3: quadrature_points[index][2],
            }), E, nu)    
    
stress.evalf(
                subs={
                    E: 200000.0,
                    nu: 0.3,
                },
            )
# %%

f_int = Matrix.zeros(24, 1)

# Note: for each dof entry we are looping over all quadrature points
for i in range(0, 8):
    B_T = calc_B(phi[i], x_1, x_2, x_3).T
    for index, w in enumerate(weights):
        
        f_int[3*i:3*i+3, 0] += w * B_T.evalf(
            subs={
                x_1: quadrature_points[index][0],
                x_2: quadrature_points[index][1],
                x_3: quadrature_points[index][2],
            }
        ) * stress[:, index]
for key in links:
    print(key)
    for i in links[key]:
        f_int.row_op(3 * key + i, lambda v, j: 0.0)

f_int_eval = f_int.evalf(
                subs={
                    E: 200000.0,
                    nu: 0.3,
                },
            )



# %%
f_int_eval+b_eval

# %%
