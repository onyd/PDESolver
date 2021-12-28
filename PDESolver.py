from GUI import PDEWindow
import numpy as np
from sympy import lambdify
from sympy.abc import x, y, t, f


def second_derivative_matrix(n, hx, hy):
    result = np.zeros(shape=(n, n))
    result[0, 0] = 1.0
    result[0, 1] = -2.0
    result[0, 2] = 1.0

    for i in range(1, n - 1):
        result[i, i - 1] = 1.0
        result[i, i] = -2.0
        result[i, i + 1] = 1.0

    result[n - 1, n - 3] = 1.0
    result[n - 1, n - 2] = -2.0
    result[n - 1, n - 1] = 1.0

    return result / hx**2, result.T / hy**2


def first_derivative_matrix(n, hx, hy):
    result = np.zeros(shape=(n, n))
    result[0, 0] = -3.0
    result[0, 1] = 4.0
    result[0, 2] = -1.0

    for i in range(1, n - 1):
        result[i, i - 1] = -1.0
        result[i, i + 1] = 1.0

    result[n - 1, n - 3] = 3.0
    result[n - 1, n - 2] = -4.0
    result[n - 1, n - 1] = 1.0

    return result / (2 * hx), result.T / (2 * hy)


def expr_to_matrix_opti(c, X, Y, time, F):
    func = lambdify((x, y, t, f), c, "numpy")
    xx, yy = np.meshgrid(X, Y)
    result = func(
        xx, yy,
        np.array([[time for i in range(len(Y))] for j in range(len(X))]), F)

    return result


def H(C3, C4, C5, C6, C7, C8, G, Axx, Ayy, Ax, Ay, F):
    return G - (C3 * np.matmul(Axx, F) + C4 * np.matmul(F, Ayy) +
                C5 * np.matmul(Ax, np.matmul(F, Ay)) + C6 * np.matmul(Ax, F) +
                C7 * np.matmul(F, Ay) + C8 * F)


def J(C1, C2, C3, C4, C5, C6, C7, C8, G, Axx, Ayy, Ax, Ay, F, dF):
    return dF, (G - C2 * dF -
                H(C3, C4, C5, C6, C7, C8, G, F, Axx, Ayy, Ax, Ay) / C1)


def update_coefficients(c1, c2, c3, c4, c5, c6, c7, c8, g, X, Y, time, F):
    global C1, C2, C3, C4, C5, C6, C7, C8, G

    if t in c1.free_symbols:
        C1 = expr_to_matrix_opti(c1, X, Y, time, F)
    if t in c2.free_symbols:
        C2 = expr_to_matrix_opti(c2, X, Y, time, F)
    if t in c3.free_symbols:
        C3 = expr_to_matrix_opti(c3, X, Y, time, F)
    if t in c4.free_symbols:
        C4 = expr_to_matrix_opti(c4, X, Y, time, F)
    if t in c5.free_symbols:
        C5 = expr_to_matrix_opti(c5, X, Y, time, F)
    if t in c6.free_symbols:
        C6 = expr_to_matrix_opti(c6, X, Y, time, F)
    if t in c7.free_symbols:
        C7 = expr_to_matrix_opti(c7, X, Y, time, F)
    if t in c8.free_symbols:
        C8 = expr_to_matrix_opti(c8, X, Y, time, F)
    if t in g.free_symbols:
        G = expr_to_matrix_opti(g, X, Y, time, F)


def init_general_solver(c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        g,
                        ht,
                        time,
                        X,
                        Y,
                        F,
                        dF=0):
    global C1, C2, C3, C4, C5, C6, C7, C8, G, Ax, Ay, Axx, Ayy

    hx = (X[-1] - X[0]) / (len(X) - 1)
    hy = (Y[-1] - Y[0]) / (len(Y) - 1)

    C1 = expr_to_matrix_opti(c1, X, Y, time, F)
    C2 = expr_to_matrix_opti(c2, X, Y, time, F)
    C3 = expr_to_matrix_opti(c3, X, Y, time, F)
    C4 = expr_to_matrix_opti(c4, X, Y, time, F)
    C5 = expr_to_matrix_opti(c5, X, Y, time, F)
    C6 = expr_to_matrix_opti(c6, X, Y, time, F)
    C7 = expr_to_matrix_opti(c7, X, Y, time, F)
    C8 = expr_to_matrix_opti(c8, X, Y, time, F)
    G = expr_to_matrix_opti(g, X, Y, time, F)

    Ax, Ay = first_derivative_matrix(len(X), hx, hy)
    Axx, Ayy = np.matmul(Ax, Ax), np.matmul(Ay, Ay)

    n_non_zero = np.count_nonzero(C1)
    # No second order time dervative
    if n_non_zero == 0:
        return (F, )
    # Second order timederivation (all non-zero coefficients)
    elif (c1.is_constant() and C1 != 0) or (not c1.is_constant()
                                            and n_non_zero == len(X) * len(Y)):
        return (F, F + ht * dF)
    else:
        raise ValueError("c1 coefficients must be zero or never zero")


def general_solver(c1,
                   c2,
                   c3,
                   c4,
                   c5,
                   c6,
                   c7,
                   c8,
                   g,
                   ht,
                   time,
                   X,
                   Y,
                   F1,
                   F0=0):
    global C1, C2, C3, C4, C5, C6, C7, C8, G, Ax, Ay, Axx, Ayy

    update_coefficients(c1, c2, c3, c4, c5, c6, c7, c8, g, X, Y, time, F1)

    n_non_zero = np.count_nonzero(C1)
    # No second order time dervative
    if n_non_zero == 0:
        return F1 + (ht / C2) * H(C3, C4, C5, C6, C7, C8, G, Axx, Ayy, Ax, Ay,
                                  F1)
    # Second order timederivation (all non-zero coefficients)
    elif (c1.is_constant() and C1 != 0) or (not c1.is_constant()
                                            and n_non_zero == len(X) * len(Y)):
        D = 2 * C1 + C2 * ht
        C = 4 * C1 / D
        return F0 * (1.0 - C) + F1 * C + (2 * ht**2 / D) * H(
            C3, C4, C5, C6, C7, C8, G, Axx, Ayy, Ax, Ay, F1)
    else:
        raise ValueError("c1 coefficients must be zero or never zero")


if __name__ == "__main__":
    window = PDEWindow([{
        "name":
        "General",
        "algo":
        general_solver,
        "init":
        init_general_solver,
        "equation":
        r"c_1\frac{\partial^2 f}{\partial t^2}+c_2\frac{\partial f}{\partial t}+c_3\frac{\partial^2 f}{\partial x^2}+c_4\frac{\partial^2 f}{\partial y^2}+c_5\frac{\partial^2 f}{\partial x\partial y}+c_6\frac{\partial f}{\partial x}+c_7\frac{\partial f}{\partial y}+c_8f=g",
        "time": {
            "step": 0.05,
            "duration": 10,
            "coefficients": [f"c_{i}" for i in range(1, 3)],
        },
        "space": {
            "x_domain": (0, 10),
            "y_domain": (0, 10),
            "coefficients": [f"c_{i}" for i in range(3, 8)],
        },
        "conditions": [r"f(x,y,0)=", r"\frac{\partial f}{\partial t}(x,y,0)="],
        "function":
        "c_8",
        "right side term":
        "g",
    }])
    window.mainloop()
