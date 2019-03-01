import functions as func
import numpy as np
np.set_printoptions(precision=3)


mat = func.read_matrix("./matrix.txt")
A = mat[:, :-1]
b = mat[:, -1]


print(f"Matrix A:\n{A}\nb column:\n {b}")
print(f"Gaussian elimination: {func.gaussian_elim(A, b)}")
print(f"Jacobi iterative method: {func.jacobi_method(A, b)}")
print(f"Gauss-Seidel iterative method: {func.gauss_seidel_method(A, b)}")
print(f"Square root method: {func.square_root_method(A, b)}")
print(f"Determinant: {func.determinant(A)}")
print(f"Inversed matrix(Square root method): \n{func.square_invert(A)}")
