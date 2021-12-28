from sympy.utilities.lambdify import lambdify
from sympy.abc import x, y
import sympy as sp
import numpy as np
import timeit as ti

expr = sp.parse_expr("1+x")

func = lambdify((x, y), expr, 'numpy')  # returns a numpy-ready function
numpy_array_of_results = func(
    np.array([[1 for i in range(5)] for j in range(5)]),
    np.array([[1 for i in range(5)] for j in range(5)]))

print(numpy_array_of_results)