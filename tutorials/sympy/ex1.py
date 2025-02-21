#%%
from sympy.vector import CoordSys3D

Sys = CoordSys3D('Sys')

O = Sys.origin

from sympy import symbols
a1, a2, a3 = symbols('a1 a2 a3')
A = O.locate_new('A', a1*Sys.i + a2*Sys.j + a3*Sys.k)

b1, b2, b3 = symbols('b1 b2 b3')
B = O.locate_new('B', b1*Sys.i + b2*Sys.j + b3*Sys.k)
c1, c2, c3 = symbols('c1 c2 c3')
C = O.locate_new('C', c1*Sys.i + c2*Sys.j + c3*Sys.k)

P = O.locate_new('P', A.position_wrt(O) + (O.position_wrt(A) / 2))

Q = A.locate_new('Q', B.position_wrt(A) / 2)
R = B.locate_new('R', C.position_wrt(B) / 2)
S = O.locate_new('R', C.position_wrt(O) / 2)

PQ = Q.position_wrt(P)
SR = R.position_wrt(S)

PQ.cross(SR)
# %%
