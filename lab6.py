import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

l = np.array([[0.238, 0.092], [0.647, 0.672], [1.316, 2.385], [2.108, 3.108], [4.892, 2.938], [6.0, 1.5]])

def l_pol(x, l_data):
    """Полином Лагранжа: Ln(x) = sum(pi(x)*f(xi) from 0 to n)"""
    pol = 1
    result = 0
    for i in range(len(l_data)):
        for j in range(len(l_data)):
            if (j != i):                
                pol *= (x - l_data[j][0]) / (l_data[i][0] - l_data[j][0])   #pi(x)
        result += pol * l_data[i][1]
        pol = 1
    return result

def n_pol(x,l_data,divs):
    p=1
    n=len(l_data)
    N=0

    for i in range(n):
        N+=divs[i]*p
        p*=(x-l_data[i][0])
    return N

def x_y_k(l_data):
    """Считает x_k, где x_k = x_0 + hk, и y_k"""
    X_k = [0] * len(l_data)
    Y_k = [0] * len(l_data)
    h = (l_data[len(l_data)-1][0] - l_data[0][0])/(len(l_data)-1)
    for i in range(len(l_data)):
        if (i != 0):
            X_k[i] = round(l_data[0][0] + (h)*i, 3)
        else: X_k[i] = l_data[0][0]
        Y_k[i] = round(l_pol(X_k[i], l), 3)
    return X_k, Y_k
    
def fin_dif(prev_fin, Y_k, n):
    """Конечная разность n-ого порядка"""
    Y = Y_k.copy()
    delta = [None] * len(Y)
    if n == 1:
        for i in range(len(Y) - 1):
            delta[i] = round(Y[i + 1] - Y[i], 4)
    else:
        for i in range(len(Y) - n):
            delta[i] = round(prev_fin[i + 1] - prev_fin[i], 4)
    return delta

def div_dif(prev_div, l_data, n):
    """Разделенная разность n-ого порядка"""
    div = [None] * len(l_data)
    if n == 1:
        for i in range(len(l_data)-1):
            div[i] = round((l_data[i + 1][1] - l_data[i][1]) / (l_data[i + 1][0] - l_data[i][0]), 4)
    else:
        for i in range(len(l_data) - n):
            div[i] = round((prev_div[i + 1] - prev_div[i]) / (l_data[i+n][0] - l_data[0][0]), 4)
    return div

def piece_lin_spline(l_data):
    """Кусочно-линейная аппроксимация"""
    A = []
    B = []
    #решаем системы уравнений
    for i in range(1, len(l_data)):
        Ai = (l_data[i][1] - l_data[i - 1][1]) / (l_data[i][0] - l_data[i - 1][0])
        Bi = l_data[i - 1][1] - Ai * l_data[i - 1][0]
        A.append(Ai)
        B.append(Bi)
    return A, B

def piece_lin_spline_func(x_value, x_points, a, b):
    interval_index = len(x_points) - 2
    for i in range(1, len(x_points)):
        if x_value < x_points[i]:
            interval_index = i - 1
            break  
    return a[interval_index] * x_value + b[interval_index]

def quadratic_spline(l_data):
    """Кусочно-квадратичная аппроксимация"""
    n = len(l_data)
    a, b, c = np.empty(n - 2), np.empty(n - 2), np.empty(n - 2)
    #решаем системы квадратных уравнений
    for i in range(0, n - 2):
        A = []
        for j in range(3):
            A.append([l_data[i+j][0] ** 2, l_data[i+j][0], 1])
        
        a[i], b[i], c[i] = np.linalg.solve(np.array(A), np.array([l_data[k][1] for k in range(i, i+3)]))
    return a, b, c

def quadratic_spline_func(x_value, x_points, a, b, c):
    interval_index = len(x_points) - 3
    for i in range(1, len(x_points) - 1):
        if x_value < x_points[i]:
            interval_index = i - 1
            break
    return c[interval_index] + (b[interval_index] + a[interval_index] * x_value) * x_value

def cubic_spline(l_data):
    """Кубический сплайн"""
    n = len(l_data)
    
    h_i = np.array([l_data[i][0] - l_data[i - 1][0] for i in range (1, n)])
    l_i = np.array([(l_data[i][1] - l_data[i - 1][1]) / h_i[i - 1] for i in range(1, n)])

    delta_i = np.empty(n - 2, float)
    lambda_i = np.empty(n - 2, float)
    delta_i[0] =  -0.5 * h_i[1] / (h_i[0] + h_i[1])
    lambda_i[0] = 1.5 * (l_i[1] - l_i[0]) / (h_i[0] + h_i[1])
    for i in range(1, n - 2):
        delta_i[i] = - h_i[i + 1] / (2 * h_i[i] + 2 * h_i[i + 1] + h_i[i] * delta_i[i - 1])
        lambda_i[i] = (2 * l_i[i + 1] - 3 * l_i[i] - h_i[i] * lambda_i[i - 1]) / \
                      (2 * h_i[i] + 2 * h_i[i + 1] + h_i[i] * delta_i[i - 1])
    
    a = np.array([l_data[i][1] for i in range(1, len(l_data))])
    b = np.empty(n - 1)
    c = np.empty(n - 1)
    d = np.empty(n - 1)
    c[n - 2] = 0
    
    for i in range(n - 3, -1, -1):
        c[i] = delta_i[i] * c[i + 1] + lambda_i[i]

    for i in range(n - 2, -1, -1):
        b[i] = l_i[i] + 2 / 3 * c[i] * h_i[i] + 1 / 3 * h_i[i] * c[i - 1]
        d[i] = (c[i] - c[i - 1]) / (3 * h_i[i])
    return a, b, c, d

def cubic_spline_func(x_value, x_p, a, b, c, d):
    i_i = len(x_p) - 2 # i_i == interval_index
    for i in range(1, len(x_p)):
        if x_value < x_p[i]:
            i_i = i - 1
            break
    return a[i_i] + (b[i_i] + (c[i_i] + d[i_i] * (x_value - x_p[i_i + 1])) * (x_value - x_p[i_i + 1])) * (x_value - x_p[i_i + 1])

if __name__ == "__main__":
    lagrange = l_pol(l[0][0] + l[1][0], l)
    print(f"\nL4(x1 + x2) = {lagrange}")

    #таблица конечных разностей
    x_y = x_y_k(l)
    fin_1 = fin_dif(None, x_y[1], 1)
    fin_2 = fin_dif(fin_1, x_y[1], 2)
    fin_3 = fin_dif(fin_2, x_y[1], 3)
    fin_4 = fin_dif(fin_3, x_y[1], 4)
    fin_5 = fin_dif(fin_4, x_y[1], 5)
    table_fin = []
    for i in range(len(l)):
        table_fin.append([x_y[0][i] ,x_y[1][i], fin_1[i], fin_2[i], fin_3[i], fin_4[i], fin_5[i]])
    print("\nFinite differences table:")
    print(tabulate(table_fin, headers=['Xk', 'Yk', '^1', '^2', '^3', '^4', '^5']))

    #таблица разделенных разностей
    div_one = div_dif(None, l, 1)
    div_two = div_dif(div_one, l, 2)
    div_three = div_dif(div_two, l, 3)
    div_four = div_dif(div_three, l, 4)
    div_five = div_dif(div_four, l, 5)
    table_div = []
    for i in range(len(l)):
        table_div.append([x_y[0][i] ,x_y[1][i], div_one[i], div_two[i], div_three[i], div_four[i], div_five[i]])
    print("\nDivided differences table:")
    print(tabulate(table_div, headers=['Xk', 'Yk', '^1', '^2', '^3', '^4', '^5']))

    diffs = [l[0][1], div_one[0], div_two[0], div_three[0], div_four[0], div_five[0]]
    print(f"\nN4(x1 + x2) = {n_pol(l[0][0] + l[1][0], l, diffs)}")
    print("\nPiecewise linear approximation: ")
    pls = piece_lin_spline(l)
    print(tabulate(zip(*pls), headers=['a', 'b']))

    print("\nPiecewise quadratic approximation: ")
    pqs = quadratic_spline(l)
    print(tabulate(zip(*pqs), headers=['a', 'b', 'c']))

    print("\nСubic spline: ")
    cs = cubic_spline(l)
    print(tabulate(zip(*cs), headers=['a', 'b', 'c', 'd'])) 
    x_data = [l[i][0] for i in range(len(l))]
    y_data = [l[i][1] for i in range(len(l))]
    range_X = np.linspace(0, x_data[-1])

    plt.figure(1)
    plt.plot(range_X, l_pol(range_X, l), label='Lagrange')
    plt.plot(range_X, n_pol(range_X, l, diffs), label='Newton')
    plt.plot(range_X, [piece_lin_spline_func(x_value, x_data, *pls) for x_value in range_X], label='Piece-linear')
    plt.plot(range_X, [quadratic_spline_func(x_value, x_data, *pqs) for x_value in range_X], label='Piece-quadratic')
    plt.plot(range_X, [cubic_spline_func(x_value, x_data, *cs) for x_value in range_X], label='Cubic')
    plt.scatter(x_data, y_data)
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(range_X, l_pol(range_X, l), label='Lagrange')
    plt.scatter(x_data, y_data)
    plt.grid()

    plt.figure(3)
    plt.plot(range_X, n_pol(range_X, l, diffs), label='Newton')
    plt.scatter(x_data, y_data)
    plt.grid()

    plt.figure(4)
    plt.plot(range_X, [piece_lin_spline_func(x_value, x_data, *pls) for x_value in range_X], label='Piece-linear')
    plt.scatter(x_data, y_data)
    plt.grid()

    plt.figure(5)
    plt.plot(range_X, [quadratic_spline_func(x_value, x_data, *pqs) for x_value in range_X], label='Piece-quadratic')
    plt.scatter(x_data, y_data)
    plt.grid()

    plt.figure(6)
    plt.plot(range_X, [cubic_spline_func(x_value, x_data, *cs) for x_value in range_X], label='Cubic')
    plt.scatter(x_data, y_data)
    plt.grid()

    plt.show()