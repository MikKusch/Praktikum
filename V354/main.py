import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



zeit, spannung = np.genfromtxt('Daten.txt', unpack = True)

def f(zeit, a, b, c):
    return a*np.exp(b*zeit)+c

params1, covariance_matrix1 = curve_fit(f, zeit[::2], spannung[::2])
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))



params2, covariance_matrix2 = curve_fit(f, zeit[1::2], spannung[1::2])
uncertainties2 = np.sqrt(np.diag(covariance_matrix2))

zeit_plot = np.linspace(zeit[0], zeit[-1], 1000)

plt.plot(zeit, spannung, '.', label = 'Messwerte')
plt.plot(zeit_plot, f(zeit_plot, *params1), label = 'Regressionkurve')

plt.xlabel(r'$\varphi/rad$')
plt.ylabel(r'$A/V$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_1.pdf')

