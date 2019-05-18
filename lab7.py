import numpy as np
import sympy as sp
from tabulate import tabulate as tbl
import matplotlib.pyplot as plt
#this entire code is some street magic

x, y = sp.symbols('x y')

int_expr = x * 3**-x
int_f = sp.lambdify(x, int_expr)
int_a = 1
int_b = 2.2

diff_expr = x * y**2 - y
diff_f = sp.lambdify((x, y), diff_expr)
diff_a = 0
diff_b = 2
y_x0 = 1
x0 = 0

def trapezoid_method(func, a, b, n):
    h = (b - a) / n
    return h * sum([(func(a + h * i) + func(a + h * (i + 1))) / 2 for i in range(n)])

def simpson_method(func, a, b, n):
    h = (b - a) / n
    return h / 3 * sum([func(a + h * i) + 4 * func(a + h * (i + 1)) + func(a + h * (i + 2)) for i  in range(0, n - 1, 2)])

def newton_int(expr, a, b):
    F = sp.integrate(expr, x)
    return F.subs(x, b) - F.subs(x, a)

def int_error(int_method, func, a, b, m_coef, eps):
    n = 1
    while True:
        if abs(int_method(func, a, b, 2 * n) - int_method(func, a, b, n)) / (2 ** m_coef - 1) < eps:
            break            
        n *= 2        
    return n * 2

def ode_error(ode_method, func, a, b, x0, y_x0, m_coef, eps):
    n = 1
    while True:
        _, y1 = ode_method(func, a, b, x0, y_x0, 2 * n)
        _, y2 = ode_method(func, a, b, x0, y_x0, n)
        if abs(y2[-1] - y1[-1]) / (2 ** 4 - 1) < eps:
            break
        n *= 2 
    return n * 2

def runge_kutta_method_4(func, a, b, x0, f_x0, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = x0
    y[0] = y_x0
    
    for i in range(n):
        F1 = h * func(x[i], y[i])
        F2 = h * func(x[i] + h / 2, y[i] + F1 / 2)
        F3 = h * func(x[i] + h / 2, y[i] + F2 / 2)
        F4 = h * func(x[i] + h, y[i] + F3)
        y[i + 1] = y[i] + 1 / 6 * (F1 + F4 + 2 * (F2 + F3))
        x[i + 1] = x[i] + h
    return x, y

def adams_method(f, a, b, x0, f_x0, n):        
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = x0
    y[0] = f_x0
    x[1] = x[0] + h
    y[1] = y[0] + h * f(x[0], y[0])
    for i in range(1, n):
        #predictor
        pred = y[i] + h / 2 * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
        x[i + 1] = x[i] + h
        #corrector
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i + 1], pred))
    return x, y

def euler_method(func, a, b, x0, f_x0, n):        
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = f_x0
    
    for i  in range(n):
        y[i + 1] = y[i] + h * func(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y

if __name__ == "__main__":
    #integrals
    int_n = int_error(trapezoid_method, int_f, int_a, int_b, 2, 0.001)
    print(f"\nintegral error est.: {int_n}")

    trap_int = trapezoid_method(int_f, int_a, int_b, int_n)
    print(f"trapezoid method with h step: {trap_int}")
    trap2_int = trapezoid_method(int_f, int_a, int_b, int_n // 2)
    print(f"...with 2h step: {trap2_int}")

    simp_int = simpson_method(int_f, int_a, int_b, int_n)
    print(f"Simpson method with h step: {simp_int}")
    simp2_int = simpson_method(int_f, int_a, int_b, int_n // 2)
    print(f"...with 2h step: {simp2_int}")

    nl_int = newton_int(int_expr, int_a, int_b)
    print(f"Newton-Leibniz formula: {nl_int}")

    sp_int = sp.integrate(int_expr, (x, int_a, int_b))
    print(f"Calculated: {sp_int}")
    
    #summary
    print(tbl(zip(
        ['Trapezoid', 'Simpson', 'Newton-Leibniz', 'Calculated'], 
        [trap_int, simp_int, nl_int, sp_int],
        [trap2_int, simp2_int, '', '']), 
        headers=['Method', 'h', '2h'], 
        tablefmt='grid', stralign='center'))

    #ODE
    diff_n = ode_error(runge_kutta_method_4, diff_f, diff_a, diff_b, x0, y_x0, 4, 0.0001)
    print(f"ODE error est.: {diff_n}")

    runge_x, runge_y = runge_kutta_method_4(diff_f, diff_a, diff_b, x0, y_x0, diff_n)
    runge_x_2, runge_y_2 = runge_kutta_method_4(diff_f, diff_a, diff_b, x0, y_x0, diff_n // 2)
    
    adams_x, adams_y = adams_method(diff_f, diff_a, diff_b, x0, y_x0, diff_n)
    adams_x_2, adams_y_2 = adams_method(diff_f, diff_a, diff_b, x0, y_x0, diff_n // 2)

    #summary
    print("Runge-Kutta method:")
    print(tbl(zip(runge_x, runge_y,
                [runge_x_2[i // 2] if i & 1 == 0 else "" for i in range(len(runge_x))],
                [runge_y_2[i // 2] if i & 1 == 0 else "" for i in range(len(runge_y))],
                [abs(runge_y_2[i // 2] - runge_y[i]) if i & 1 == 0 else "" for i in range(len(runge_x))]),
                headers=['x_i','y_i', '~x_i', '~y_i', 'delta_i'],
                tablefmt='grid', stralign='center'))
    
    print("Adams method:")
    print(tbl(zip(adams_x, adams_y,
                [adams_x_2[i // 2] if i & 1 == 0 else "" for i in range(len(adams_x))],
                [adams_y_2[i // 2] if i & 1 == 0 else "" for i in range(len(adams_y))],
                [abs(adams_y_2[i // 2] - adams_y[i]) if i & 1 == 0 else "" for i in range(len(adams_x))]), 
                headers=['x_i','y_i', '~x_i', '~y_i', 'delta_i'],
                tablefmt='grid', stralign='center'))

    euler_x, euler_y = euler_method(diff_f, diff_a, diff_b, x0, y_x0, diff_n)
    euler_x_2, euler_y_2 = euler_method(diff_f, diff_a, diff_b, x0, y_x0, diff_n // 2)

    print("Euler method:")
    print(tbl(zip(euler_x, euler_y,
                [euler_x_2[i // 2] if i & 1 == 0 else "" for i in range(len(euler_x))],
                [euler_y_2[i // 2] if i & 1 == 0 else "" for i in range(len(euler_y))],
                [abs(euler_y_2[i // 2] - euler_y[i]) if i & 1 == 0 else "" for i in range(len(euler_x))]),
                headers=['x_i','y_i', '~x_i', '~y_i', 'delta_i'],
                tablefmt='grid', stralign='center'))

    f = sp.Function('f')
    diff_solution = sp.dsolve(sp.Eq(sp.diff(f(x), x) - x * f(x)**2 + f(x)), f(x))
    print(f"ODE solution: {diff_solution}")
    # C1 = sp.solve(np.exp(x0)/(x + (1-x0)*np.exp(x0)), x)
    # print(C1)
    func_solution = sp.lambdify(x, sp.exp(-x)/((1 + x) * sp.exp(-x)))

    solved_y = np.array([func_solution(xi) for xi in runge_x])
    print(tbl(zip(["{0:0.5f}".format(i) for i in runge_x],
        ["{0:0.5f}".format(i) for i in solved_y],
        ["{0:0.5f}".format(i) for i in runge_y],
        [abs(runge_y[i] - solved_y[i]) for i in range(len(runge_y))],
        ["{0:0.5f}".format(i) for i in adams_y],
        ["{0:0.5f}".format(abs(adams_y[i] - solved_y[i])) for i in range(len(adams_y))]),
        headers=['x_i', 'Solution | y_i', 'Runge-Kutta | y_i', 'deltaR_i','Adams | y_i', 'deltaA_i'],
        tablefmt='grid', stralign='center'))


    # plot street magic
    plt.figure(1)
    plt.plot(runge_x, runge_y, label='h')
    plt.plot(runge_x_2, runge_y_2, label='2h')
    plt.title('Runge-Kutta method')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(adams_x, adams_y, label='h')
    plt.plot(adams_x_2, adams_y_2, label='2h')
    plt.title('Adams method')
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(euler_x, euler_y, label='h')
    plt.plot(euler_x_2, euler_y_2, label='2h')
    plt.title('Euler')
    plt.legend()
    plt.grid()

    plt.figure(4)
    range_x = np.linspace(diff_a, diff_b)
    plt.plot(range_x, func_solution(range_x))
    plt.title('Solution')
    plt.grid()

    plt.figure(5)
    plt.plot(runge_x, runge_y, label='Runge-Kutta')
    plt.plot(adams_x, adams_y, label='Adams')
    plt.plot(euler_x, euler_y, label='Euler')
    plt.plot(range_x, func_solution(range_x), label='Solution')
    plt.legend()
    plt.grid()
    
    plt.show()
