from sympy import symbols, Matrix, diff, Function, pprint, Symbol
from sympy import init_printing
init_printing()  # Enable pretty printing

x_1, x_2, x_3 = symbols("x_1 x_2 x_3")

phi_i = Function('phi_i')(x_1, x_2, x_3)
phi_j = Function('phi_j')(x_1, x_2, x_3)


B = Matrix.zeros(6, 3)
B[0, 0] = diff(phi_i, x_1)
B[1, 1] = diff(phi_i, x_2)
B[2, 2] = diff(phi_i, x_3)

B[3, 1] = diff(phi_i, x_3)
B[3, 2] = diff(phi_i, x_2)

B[4, 0] = diff(phi_i, x_3)
B[4, 2] = diff(phi_i, x_1)

B[5, 0] = diff(phi_i, x_2)
B[5, 1] = diff(phi_i, x_1)
B

# Define a 6x6 matrix of symbolic coefficients
# Define symbolic material properties
E, nu = symbols('E nu')

# Compute Lamé parameters
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Define the 6x6 elastic stiffness matrix for isotropic material
C = Matrix([
    [lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
    [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
    [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
    [0, 0, 0, mu, 0, 0],
    [0, 0, 0, 0, mu, 0],
    [0, 0, 0, 0, 0, mu]
])
# Display the matrix
#pprint(C)

B.T*C

BT = Matrix.zeros(6, 3)
BT[0, 0] = diff(phi_j, x_1)
BT[1, 1] = diff(phi_j, x_2)
BT[2, 2] = diff(phi_j, x_3)

BT[3, 1] = diff(phi_j, x_3)
BT[3, 2] = diff(phi_j, x_2)

BT[4, 0] = diff(phi_j, x_3)
BT[4, 2] = diff(phi_j, x_1)

BT[5, 0] = diff(phi_j, x_2)
BT[5, 1] = diff(phi_j, x_1)
K = B.T*C*BT
K
#print(K.is_symmetric())