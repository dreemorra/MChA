import matplotlib.pyplot as plt
import numpy as np

#график уравнения
plt.figure(1)
#данные
tx = np.linspace(-1, 50, 10000)
plt.axis([-10, 50, -50, 10])
#оси координат
plt.axhline(linewidth=1, color='red')
plt.axvline(linewidth=1, color='red')
plt.xlabel('f(x) = 1 - x + sin(x) - ln(1+x)')

plt.plot(tx, 1 - tx + np.sin(tx) - np.log1p(tx))

#график системы уравнений
plt.figure(2)
x, y = np.linspace(-10,10,1000), np.linspace(-10,10,1000)
X, Y = np.meshgrid(x,y)
#оси координат
plt.axhline(linewidth=1, color='red')
plt.axvline(linewidth=1, color='red')
#графики
plt.contour(X, Y, np.tan(X*Y) - X**2, [0])
plt.contour(X, Y, 0.5*(X**2) + 3*(Y**2) - 1, [0], colors='green')
plt.show()