# %%
import os
os.chdir("/home/musa/Documents/main_projects/my_demos/solid_mechanics_low_level/manual_implementation") 

from sympy import symbols, Matrix, diff
from utils import constitutive, getBasisFunctions, calc_B, assemble_matrix, assemble_vector, set_bc, getStrainExpr
import basix


quadrature_points, weights = basix.make_quadrature(basix.CellType.hexahedron, 2)

x_1, x_2, x_3, phi_1, E, nu, u_1, bf = symbols("x_1 x_2 x_3 phi_1 E nu u_1 bf")

phi = getBasisFunctions(x_1, x_2, x_3)
# %%
du = Matrix.zeros(24, 1)
u = Matrix.zeros(24, 1)
B = Matrix.zeros(6, 3)
A = Matrix.zeros(24, 24)

# %%
stress = Matrix.zeros(6, len(weights))
dstrain = Matrix.zeros(6, len(weights))
#tangent, stressIn = constitutive(stressIn, dstrain, E, nu)

strain = getStrainExpr(du, phi, x_1, x_2, x_3)
for index, w in enumerate(weights):
    tangent, stress[:, index] = constitutive(stress[:, index], strain.evalf(
            subs={
                x_1: quadrature_points[index][0],
                x_2: quadrature_points[index][1],
                x_3: quadrature_points[index][2],
            }), E=200000, nu=0.3)    

A = assemble_matrix(A, phi, x_1, x_2, x_3, E, nu, weights, quadrature_points, tangent)        

# %%

bcs = {0: [0, 1, 2], 1: [1, 2], 2: [0, 2], 3: [2], 4: [0, 1], 5: [1], 6: [0]}

# %%
body_force = Matrix([-1000, 0.0, 0.0])
b = Matrix.zeros(24, 1)



b = assemble_vector(b, phi, x_1, x_2, x_3, body_force, weights, quadrature_points, stress)
# %%
A, b = set_bc(A, b, bcs)
#b
# %%
du = A.LUsolve(b)
#du
u = u - du
#u

# %%
strain = getStrainExpr(du, phi, x_1, x_2, x_3)
for index, w in enumerate(weights):
    tangent, stress[:, index] = constitutive(stress[:, index], strain.evalf(
            subs={
                x_1: quadrature_points[index][0],
                x_2: quadrature_points[index][1],
                x_3: quadrature_points[index][2],
            }), E=200000, nu=0.3) 

stress

