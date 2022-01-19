import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t_2, p_a, leist, t_1, p_b = np.genfromtxt('V206.txt', unpack = True)
p_2 = np.log(p_a)
t_2 = 1/(t_2+273.15)

p_1 = np.log(p_b)
t_1 = 1/(t_1+273.15)

def l(t, m, b):
    return -m*t+b

params, covariance_matrix = curve_fit(l, t_2, p_2)
uncertainties1 = np.sqrt(np.diag(covariance_matrix))

params_2, covariance_matrix_2 = curve_fit(l, t_1, p_1)
uncertainties2 = np.sqrt(np.diag(covariance_matrix_2))

temp_plot = np.linspace(t_2[0], t_2[-1], 1000)
temp_plot_2 = np.linspace(t_1[0], t_1[-1], 1000)


print(*params)
print(*params_2)


plt.plot(t_2, p_2, '.', label = 'Messwerte')
plt.plot(t_1, p_1, '.', label = 'Messwerte')

plt.plot(temp_plot, l(temp_plot, *params), label = 'Regressionskurve')
plt.plot(temp_plot_2, l(temp_plot_2, *params_2), label = 'Regressionskurve')

plt.xlabel(r'1/T in $K^{-1}$')
plt.ylabel(r'$log(p/p_0)$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_3.pdf')
