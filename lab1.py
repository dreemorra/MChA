import numpy as np
import functions as func

np.set_printoptions(precision=3)

mat = func.read_matrix("./matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]

if np.linalg.det(A) != 0:
    x = func.gaussian_elim(A, b)
    inv_a = func.gauss_invert(A)

    abs_b = 0.001
    rel_b = abs_b/func.norm(b)

    abs_x = func.norm(inv_a)*abs_b
    rel_x = abs_x/func.norm(x)

    print(f"Matrix A:\n{A}\nb column:\n {b}\nSolution:\n {x}")
    print(f"Inversed matrix:\n{inv_a}")
    print(f"Absolute error ∆x: {abs_x}\nRelative error δx: {rel_x}")
    #print(np.linalg.solve(A, b))                             #для проверки правильности алгоритма с помощью numpy
    #print(np.linalg.inv(A))                                  #для проверки правильности алгоритма вычисления обратной матрицы с помощью numpy