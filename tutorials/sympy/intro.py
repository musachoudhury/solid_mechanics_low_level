#%%

import sympy

sympy.sqrt(2)
# %%
from sympy import symbols

x, y = symbols('x y')
expr = x + 2*y
expr
# %%
expr + 1
#%%
expr - x
# %%
x*(x + 2*y)
# %%
from sympy import expand, factor
expanded_expr = expand(x*expr)
factor(expanded_expr)
# %%
from sympy import *
x, t, z, nu = symbols('x t z nu')
init_printing(use_unicode=True)
# %%
diff(sin(x)*exp(x), x)
# %%
integrate(exp(x)*sin(x) + exp(x)*cos(x), x)
# %%
integrate(sin(x**2), (x, -oo, oo))
# %%
y = Function('y')
dsolve(Eq(y(t).diff(t, t) - y(t), exp(t)), y(t))
# %%
Matrix([[1, 2], [2, 2]]).eigenvals()
# %%
besselj(nu, z).rewrite(jn)
# %%
latex(Integral(cos(x)**2, (x, 0, pi)))

# %%