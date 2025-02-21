#%%
from sympy.vector import CoordSys3D

N = CoordSys3D('N')

#%%
N.i
type(N.i)
# %%
v = 2*N.i + N.j
type(v)
# %%
v - N.j
type(v - N.j)
# %%
from sympy.vector import Vector
Vector.zero
# %%
type(Vector.zero)
# %%
N.i + Vector.zero
# %%
