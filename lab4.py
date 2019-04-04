import functions as func
import numpy as np
np.set_printoptions(precision=3)

mat = func.read_matrix("./matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]

Jacobi = func.jacobi_eigenvalue(A)
print(f"Matrix A:\n{A}\nb column:\n {b}\n")
print(f"Jacobi eigenvalues: {Jacobi[0]}\n")
print(f"Jacobi eigenvectors: \n{Jacobi[1]}")
print(np.linalg.eig(func.matrix_multiplication(func.transp_mat(A), A)))
