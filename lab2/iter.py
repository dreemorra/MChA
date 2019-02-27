import numpy as np
import math
np.set_printoptions(precision=3)

def read_matrix(fname: str) -> np.array:                    #чтение матрицы из файла
    matrix = []
    with open("./matrix.txt", 'r') as file_mat:
        for line in file_mat.readlines():
            matrix.append(line.strip().split(" "))
    matrix = np.array(matrix, dtype = float)
    return matrix

def norm(arr: np.array):                                    #нахождение нормы матриц
    return max([np.sum(abs(line)) for line in arr])

def apply_mat(A: np.array, x: np.array):
    return np.array([np.sum(x*A[i]) for i in range(len(A))])

def iter(B: np.array, b: np.array):
    B = B.copy()
    b = b.copy()
    x0 = np.array([1/2 for i in range(len(b))])
    x = apply_mat(B, x0) + b
    k = math.floor(math.log(0.01/(norm(x-x0))*(1-norm(B)), norm(B)))
    for i in range(100500):
        x = apply_mat(B, x) + b
    return x

mat = read_matrix("/home/mora/matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]
E = np.identity(len(A))

for i in range(0, len(A)):
    A[i]/=-A[i,i]
    b[i]/=A[i,i]
    A[i,i]=0
B = A
if (norm(B) < 1):
    print(iter(B, b))