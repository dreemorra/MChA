import numpy as np
np.set_printoptions(precision=3)

def gauss_solve(A: np.array, b: np.array) -> np.array:      #решение матрицы методом Гаусса
    A = A.copy()
    b = b.copy()
    for i in range(0, len(A)):                              #прямой ход
        b[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(i+1, len(A)):
            b[j] -= b[i]*A[j, i]
            A[j] -= A[i]*A[j, i]
    x = b
    for k in range(len(A)-1, -1, -1):                       #обратный ход
        for m in range(len(A)-1, k, -1):
            x[k] -= (A[k, m]*x[m])/A[k, k]
    return x


def gauss_invert(A: np.array) -> np.array:                  #нахождение обратной матрицы методом Гаусса
    A = A.copy()
    E = np.identity(len(A))                                 #единичная матрица
    for i in range(0, len(A)):                              #прямой и обратный ход
        E[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(0, len(A)):
            if j != i:
                E[j] -= E[i]*A[j, i]
                A[j] -= A[i]*A[j, i]
    return E


def read_matrix(fname: str) -> np.array:                    #чтение матрицы из файла
    matrix = []
    with open("./matrix.txt", 'r') as file_mat:
        for line in file_mat.readlines():
            matrix.append(line.strip().split(" "))
    matrix = np.array(matrix, dtype = float)
    return matrix

def norm(arr: np.array):                                    #нахождение нормы матриц
    return max([np.sum(abs(line)) for line in arr])

mat = read_matrix("/home/mora/matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]
if np.linalg.det(A) != 0:
    x = gauss_solve(A, b)
    inv_a = gauss_invert(A)

    abs_b = 0.001
    rel_b = abs_b/norm(b)

    abs_x = norm(inv_a)*abs_b
    rel_x = abs_x/norm(x)

    print(f"Matrix A:\n{A}\nb column:\n {b}\nSolution:\n {x}")
    print(f"Inversed matrix:\n{inv_a}")
    print(f"Absolute error ∆x: {abs_x}\nRelative error δx: {rel_x}")
    #print(np.linalg.solve(A, b))                             #для проверки правильности алгоритма с помощью numpy
    #print(np.linalg.inv(A))                                  #для проверки правильности алгоритма вычисления обратной матрицы с помощью numpy