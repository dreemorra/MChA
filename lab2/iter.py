import numpy as np
import math
np.set_printoptions(precision=3)

#чтение матрицы из файла
def read_matrix(fname: str) -> np.array:  
    matrix = []
    with open(fname, 'r') as file_mat:
        for line in file_mat.readlines():
            matrix.append(line.strip().split(" "))
    matrix = np.array(matrix, dtype=float)
    return matrix

# нахождение нормы матрицы
def norm(arr: np.array):  
    return max([np.sum(abs(line)) for line in arr])

# умножение матрицы и вектора
def apply_mat(A: np.array, x: np.array):  
    return np.array([np.sum(x*A[i]) for i in range(len(A))])

# решение системы методом Гаусса
def gaussian_elim(A: np.array, b: np.array) -> np.array:  
    A = A.copy()
    b = b.copy()
    for i in range(0, len(A)):  # прямой ход
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
    for i in range(0, len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A
    if norm(B) < 1:
        x0 = np.array([1/2 for i in range(len(b))])
        x = apply_mat(B, x0) + b
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        for i in range(0, k+1):
            x = apply_mat(B, x) + b
        return x

# решение системы методом Гаусса-Зейделя
def gauss_seidel_method(A: np.array, b: np.array) -> np.array:
    A = A.copy()
    b = b.copy()
    E = np.identity(len(A))
    for i in range(0, len(A)):
        b[i] /= A[i, i]
        A[i] /= A[i, i]
    B = E - A
    if np.sum([abs(B[i,i]) > np.sum(B[i]) - B[i,i] for i in range(len(A))]):
        x0 = np.array([1/2 for i in range(len(b))])
        x = np.array(apply_mat(B, x0) + b)
        k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
        for n in range(k-1):
            for i in range(len(A)):
                x[i] = np.sum(B[i, :i]*x[:i]) + np.sum(B[i, i+1:len(A)]*x0[i+1:len(A)]) + b[i]
            x0 = x
        return x

mat = read_matrix("./matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]


print(f"Matrix A:\n{A}\nb column:\n {b}")
print(f"Gaussian elimination: {gaussian_elim(A, b)}")
print(f"Jacobi iterative method: {jacobi_method(A, b)}")
print(f"Gauss-Seidel iterative method: {gauss_seidel_method(A, b)}")