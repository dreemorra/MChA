import matplotlib.pyplot as plt
import numpy as np
import functions as func

def f(x):
    return 1 - x + np.sin(x) - np.log1p(x)

def der_f(x):
    return np.cos(x) - 1 - 1/(x+1)

def jacobian(x, y):
    J = [[0,0], [0,0]]
    J[0][0] = -2*x + y/((np.cos(x*y))**2)
    J[0][1] = x/((np.cos(x*y))**2)
    J[1][0] = x
    J[1][1] = 6 * y
    return J

def chords_method():
    """Метод хорд для решения линейного уравнения."""

    a = 0
    b = 5
    dif = 1

    x = a
    k = 0
    while (dif > 0.001):
        X = x - f(x) * (b - x) / (f(b) - f(x))
        dif = np.abs(x - X)
        x = X
        k+=1
    return x, k

def tan_method():
    """Метод касательных для решения линейного уравнения."""

    a = 0
    dif = 1

    x = a
    k = 0
    while (dif > 0.001):
        X = x - f(x) / der_f(x)
        dif = np.abs(x - X)
        x = X
        k+=1
    return x, k

def iter_method():
    """Метод простых итераций для системы уравнений."""

    x = 0.4
    y = 0.3
    dif = 1

    k = 0
    X = x
    Y = y
    while (dif > 0.001):
        y = np.sqrt((1-0.5*(Y**2))/3)
        x = np.sqrt(np.tan(X*Y))
        dif = np.abs(x - X)
        X = x
        Y = y
        k+=1
    return x, y, k

def newton_method(is_modified = False):
    """Метод Ньютона для решения систем уравнений.
    При is_modified = True будет использоваться модифицированная версия метода."""

    k = 0
    x = -0.4
    y = -0.3
    dif = 1

    if is_modified == True:
        J = np.linalg.inv(jacobian(x, y))
    while (dif > 0.001):
        if is_modified == False:
            J = np.linalg.inv(jacobian(x, y))
        #F
        mas = [np.tan(x*y) - x**2, 0.5*(x**2) + 3*(y**2) - 1]
        #J^-1 * F
        res = func.apply_mat(J, mas)
        X = x
        x -= res[0]
        y -= res[1]
        dif = np.abs(x - X)
        k+=1
    return x, y, k

if __name__ == "__main__":
    chord_res = chords_method()
    tan_res = tan_method()
    iter_res = iter_method()
    newton_res = newton_method()
    mod_res = newton_method(True)
    print(f"Chords method: x = {chord_res[0]} with {chord_res[1]} steps")
    print(f"Tangential method: x = {tan_res[0]} with {tan_res[1]} steps")
    print(f"Iteration method(system of eqs): x = {iter_res[0]}, y = {iter_res[1]} with {iter_res[2]} steps")
    print(f"Newton method(system of eqs): x = {newton_res[0]}, y = {newton_res[1]} with {newton_res[2]} steps")
    print(f"Modified Newton method(system of eqs): x = {mod_res[0]}, y = {mod_res[1]} with {mod_res[2]} steps")
    