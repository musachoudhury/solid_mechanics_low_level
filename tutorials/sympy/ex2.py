#%%
from sympy.vector import CoordSys3D, Del

delop = Del()

C = CoordSys3D('C')

#%%
from sympy import symbols, Function

v1, v2, v3, f = symbols('v1 v2 v3 f', cls=Function)

vfield = v1(C.x, C.y, C.z)*C.i + v2(C.x, C.y, C.z)*C.j + v3(C.x, C.y, C.z)*C.k

ffield = f(C.x, C.y, C.z)

lhs = (delop.dot(ffield * vfield)).doit()

rhs = ((vfield.dot(delop(ffield))) + (ffield * (delop.dot(vfield)))).doit()

lhs.expand().simplify() == rhs.expand().doit().simplify()
# %%
