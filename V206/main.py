import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t_2, p_a, leist, t_1, p_b = np.genfromtxt('V206.txt', unpack = True)
zeit = np.linspace(0, 840, 28)

def f (zeit, a, b, c):
    return a*zeit**2+b*zeit+c


params, covariance_matrix = curve_fit(f, zeit, t_1)

print(*params)

temp_plot = np.linspace(zeit[0], zeit[-1], 1000)

plt.plot(zeit, t_1, '.', label = 'Messwerte')
plt.plot(temp_plot, f(temp_plot, *params), label = 'Regressionskurve')

plt.xlabel(r't in s')
plt.ylabel(r'$T_1 in K$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_1.pdf')


