import numpy as np
import math

#чтение матрицы из файла
def read_matrix(fname: str) -> np.array:
    matrix = []
    with open(fname, 'r') as file_mat:
        for line in file_mat.readlines():
            matrix.append(line.strip().split(" "))
    matrix = np.array(matrix, dtype=float)
    return matrix

#умножение матриц
def matrix_multiplication(m1: np.array, m2: np.array):
    m1 = m1.copy()
    m2 = m2.copy()
    return np.array([[sum(a*b for a,b in zip(X_row, Y_col)) for Y_col in zip(*m2)] for X_row in m1])

# нахождение нормы матрицы
def norm(arr: np.array):
    return max([np.sum(abs(line)) for line in arr])

# умножение матрицы и вектора
def apply_mat(A: np.array, x: np.array):
    return np.array([np.sum(x*A[i]) for i in range(len(A))])

# нахождение обратной матрицы методом Гаусса
def gauss_invert(A: np.array) -> np.array:
    A = A.copy()
    E = np.identity(len(A))                                 #единичная матрица
    for i in range(len(A)):                              #прямой и обратный ход
        E[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(len(A)):
            if j != i:
                E[j] -= E[i]*A[j, i]
                A[j] -= A[i]*A[j, i]
    return E

#обратная матрица методом квадратного корня
def square_invert(A: np.array) -> np.array:
    A = A.copy()
    E = np.identity(len(A))
    inv_A = np.zeros(A.shape)
    for i in range(len(A)):
        inv_A[i] = square_root_method(A, E[i], True)
    return inv_A

# транспонирование матрицы
def transp_mat(A: np.array) -> np.array:
    A = A.copy()
    for i in range(len(A)):
        for j in range(i+1, len(A)):
                A[i, j], A[j, i] = A[j, i], A[i, j]
    return A

# определитель матрицы
def determinant(A: np.array) -> np.array:
    A = A.copy()
    A = matrix_multiplication(transp_mat(A), A)   #симметризация
    U = np.zeros(A.shape)

    for i in range(len(A)):
        s = A[i,i]

        for k in range(i):
            s-= U[k, i]**2

        U[i, i] = math.sqrt(s)

        for j in range(i+1, len(A)):
            s = A[i, j]
            
            for k in range(i):
                s-= U[k, i]*U[k, j]

            U[i, j] =  s/U[i, i]

    det = 1
    for i in range(len(U)):
        det *= U[i][i] ** 2
    return det

# решение системы методом Гаусса
def gaussian_elim(A: np.array, b: np.array) -> np.array:
    A = A.copy()
    b = b.copy()
    for i in range(len(A)):  # прямой ход
        b[i] /= A[i, i]
        A[i] /= A[i, i]
        for j in range(i+1, len(A)):
            b[j] -= b[i]*A[j, i]
            A[j] -= A[i]*A[j, i]
    x = b
    for k in range(len(A)-1, -1, -1):  # обратный ход
        for m in range(len(A)-1, k, -1):
            x[k] -= (A[k, m]*x[m])/A[k, k]
    return x

# решение системы методом простых итераций(метод Якоби)
def jacobi_method(A: np.array, b: np.array) -> np.array:
    A = A.copy()
    b = b.copy()
    E = np.identity(len(A))
    for i in range(len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A
    if norm(B) < 1:
        x0 = np.array([1/2 for i in range(len(b))])
        x = apply_mat(B, x0) + b
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        print(f"Jacobi iterations: {k}")
        for i in range(k+1):
            x = apply_mat(B, x) + b
        return x

# решение системы методом Гаусса-Зейделя
def gauss_seidel_method(A: np.array, b: np.array) -> np.array:
    A = A.copy()
    b = b.copy()
    E = np.identity(len(A))
    for i in range(len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A
    if np.sum([abs(B[i,i]) > np.sum(B[i]) - B[i,i] for i in range(len(A))]):
        x0 = b
        x = np.array(apply_mat(B, x0) + b)
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        print(f"Seidel iterations: {k+1}")
        for i in range(k+1):
            for i in range(len(A)):
                x[i] = np.sum(B[i, :i]*x[:i]) + np.sum(B[i, i+1:len(A)]*x0[i+1:len(A)]) + b[i]
            x0 = x
        return x

# решение системы методом квадратного корня
def square_root_method(A: np.array, b: np.array, isInverse: bool = 0) -> np.array:
    A = A.copy()
    b = b.copy()
    if isInverse == 0:
        b = apply_mat(transp_mat(A), b)
    A = matrix_multiplication(transp_mat(A), A)   #симметризация
    #print(f"Симметризированная A:\n{A}")
    #print(f"Симметризированная b:\n{b}")
    U = np.zeros(A.shape)

    for i in range(len(A)):
        s = A[i,i]
        for k in range(i):
            s-= U[k, i]**2
        U[i, i] = math.sqrt(s)

        for j in range(i+1, len(A)):
            s = A[i, j]
            for k in range(i):
                s-= U[k, i]*U[k, j]
            U[i, j] =  s/U[i, i]

    y = np.zeros(b.shape)
    x = np.zeros(b.shape)
    Ut = transp_mat(U)

    for i in range(len(y)):
        s = b[i]
        for k in range(i):
            s-=Ut[i, k]*y[k]
        y[i] = s/Ut[i, i]

    for i in range(len(y)-1, -1, -1):
        s = y[i]
        for k in range(i+1, len(A)):
            s-=U[i, k]*x[k]
        x[i] = s/U[i,i]
    return x