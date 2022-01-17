import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t_2, p_a, leist, t_1, p_b = np.genfromtxt('V206.txt', unpack = True)
zeit = np.linspace(0, 840, 28)


def g(zeit, a, b, c):
    return a/(1+zeit**c*b)


params_2, covariance_matrix_2 = curve_fit(g, zeit, t_2)

temp_plot = np.linspace(zeit[0], zeit[-1], 1000)

print(*params_2)

plt.plot(zeit, t_2, '.', label = 'Messwerte')
plt.plot(temp_plot, g(temp_plot, *params_2), label = 'Regressionskurve')

plt.xlabel(r't in s')
plt.ylabel(r'$T_2 \, in \, K$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
