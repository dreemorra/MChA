import numpy as np
import math

def read_matrix(fname: str) -> np.array:
    """Чтение матрицы из файла."""
    matrix = []
    with open(fname, 'r') as file_mat:
        for line in file_mat.readlines():
            matrix.append(line.strip().split(" "))
    matrix = np.array(matrix, dtype=float)
    return matrix

def matrix_multiplication(m1: np.array, m2: np.array):
    """Нахождение произведения матриц: m1*m2."""
    m1 = m1.copy()
    m2 = m2.copy()
    return np.array([[sum(a*b for a,b in zip(X_row, Y_col)) for Y_col in zip(*m2)] for X_row in m1])

def norm(arr: np.array):
    """Нахождение нормы матрицы."""
    return max([np.sum(abs(line)) for line in arr])

def apply_mat(A: np.array, x: np.array):
    """Перемножение матрицы и вектора."""
    return np.array([np.sum(x*A[i]) for i in range(len(A))])

def gauss_invert(A: np.array) -> np.array:
    """Вычисление обратной матрицы методом Гаусса."""

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

def square_invert(A: np.array) -> np.array:
    """Вычисление обратной матрицы методом квадратного корня."""

    A = A.copy()
    E = np.identity(len(A))
    inv_A = np.zeros(A.shape)
    for i in range(len(A)):
        inv_A[i] = square_root_method(A, E[i], True)
    return inv_A

def transp_mat(A: np.array) -> np.array:
    """Транспонирование матрицы А."""
    A = A.copy()
    for i in range(len(A)):
        for j in range(i+1, len(A)):
                A[i, j], A[j, i] = A[j, i], A[i, j]
    return A

def determinant(A: np.array) -> np.array:
    """Вычисление определителя симмметризированной матрицы"""
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

def gaussian_elim(A: np.array, b: np.array) -> np.array:
    """Решение системы методом Гаусса."""
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

def jacobi_method(A: np.array, b: np.array) -> np.array:
    """Решение системы методом простых итераций(метод Якоби)"""

    A = A.copy()
    b = b.copy()
    E = np.identity(len(A))

    #преобразуем к виду x = (E-A)x + b или x = Bx + b
    for i in range(len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A

    if norm(B) < 1:
        x0 = np.array([1/2 for i in range(len(b))])
        x = apply_mat(B, x0) + b
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        print(f"Jacobi iterations: {k+1}")
        #x_k+1 = B*x_k + b
        for i in range(k+1):
            x = apply_mat(B, x) + b
        return x

def gauss_seidel_method(A: np.array, b: np.array) -> np.array:
    """Решение системы методом Гаусса-Зейделя."""
    A = A.copy()
    b = b.copy()
    E = np.identity(len(A))
    #преобразуем к виду x = (E-A)x + b или x = Bx + b
    for i in range(len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A

    if np.sum([abs(B[i,i]) > np.sum(B[i]) - B[i,i] for i in range(len(A))]):
        x0 = b
        x = np.array(apply_mat(B, x0) + b)
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        print(f"Seidel iterations: {k+1}")
        while norm(x - x0) > 0.01:
            x0 = x
            #x[i]_k+1 = b[i] - sum(a[i,j]*x[j]_(k+1) from j = 1, j!=i to i-1) - sum(a[i,j]*x[j]_k from j = i+1 to n)
            for i in range(len(A)):
                x[i] = np.sum(B[i, :i]*x[:i]) + np.sum(B[i, i+1:len(A)]*x0[i+1:len(A)]) + b[i]
        return x

def square_root_method(A: np.array, b: np.array, isInverse: bool = 0) -> np.array:
    """Решение системы методом квадратного корня."""
    A = A.copy()
    b = b.copy()
    #симметризация
    if isInverse == 0:
        b = apply_mat(transp_mat(A), b)
    A = matrix_multiplication(transp_mat(A), A)  
    U = np.zeros(A.shape)

    for i in range(len(A)):
        #U[i,i] = sqrt(a[i,i] - sum(u[k, i]**2 for k from 1 to i-1))
        s = A[i,i]
        for k in range(i):
            s-= U[k, i]**2
        U[i, i] = math.sqrt(s)

        #U[i, j] = (a[i,j] - sum(u[k,i]*u[k, j] for k from 1 to i-1), j = i+1, n) / u[i,i]
        for j in range(i+1, len(A)):
            s = A[i, j]
            for k in range(i):
                s-= U[k, i]*U[k, j]
            U[i, j] =  s/U[i, i]

    y = np.zeros(b.shape)
    x = np.zeros(b.shape)
    Ut = transp_mat(U)
    #решаем систему Ut*y = B
    for i in range(len(y)):
        s = b[i]
        for k in range(i):
            s-=Ut[i, k]*y[k]
        y[i] = s/Ut[i, i]
    #решаем систему U*x = y
    for i in range(len(y)-1, -1, -1):
        s = y[i]
        for k in range(i+1, len(A)):
            s-=U[i, k]*x[k]
        x[i] = s/U[i,i]
    return x

def max_elem(A: np.array):
    """Поиск максимального элемента в матрице А а[i0][j0], где i0 < j0"""
    max_num = 0.0
    a = 0
    b = 0
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if math.fabs(A[i, j]) >= max_num:
                max_num = math.fabs(A[i, j])
                a, b = i, j
    return max_num, a, b

def jacobi_eigenvalue(A: np.array):
    """Вычисление собственных значений матрицы."""
    A = A.copy()
    UVectors = np.identity(len(A))
    #симметризация
    A = matrix_multiplication(transp_mat(A), A)

    sumOfElements = 1
    while sumOfElements > 0.001:
        #макс. элемент
        kek = max_elem(A)
        #считается угол f, такой, чтобы у матрицы (A(новая) = UT*A*U) a[i][j] обращался в нуль
        f = math.atan(2*A[kek[1], kek[2]]/(A[kek[1], kek[1]] - A[kek[2], kek[2]])) / 2
        #создается единичная матрица и матрица поворота
        U = np.identity(len(A))
        U[kek[1], kek[1]], U[kek[2], kek[2]] = math.cos(f), math.cos(f)
        U[kek[2], kek[1]] = math.sin(f)
        U[kek[1], kek[2]] = -1*math.sin(f)
        #собственные векторы U = U[0]*U[1]*...*U[k-2]*U[k-1]
        UVectors = matrix_multiplication(UVectors, U)
        #A(новая) = UT*A*U
        A = transp_mat(U) @ A @ U
        #точность
        sumOfElements = 0
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    sumOfElements += A[i][j] ** 2
    return np.array([A[i,i] for i in range(len(A))]), np.array(UVectors)
