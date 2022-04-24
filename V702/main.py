import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



imp_rho = np.genfromtxt('Daten_Rhod.txt', unpack = True)
imp_van = np.genfromtxt('Daten_Vana.txt', unpack = True)

imp_rho = imp_rho-4.75
imp_van = imp_van-9.5

t_rho = np.linspace(0, 750, 50)
t_van = np.linspace(0, 900, 30)


def N(t, b, λ):
    return λ*t+b



params_rho, covariance_matrix_rho = curve_fit(N, t_rho, np.log(imp_rho))
uncertainties_rho = np.sqrt(np.diag(covariance_matrix_rho))

params_van, covariance_matrix_van = curve_fit(N, t_van, np.log(imp_van))
uncertainties_van = np.sqrt(np.diag(covariance_matrix_van))





t_plot_rho = np.linspace(0, 750, 1000)
t_plot_van = np.linspace(0, 900, 1000)


plt.semilogy(t_rho, np.log(imp_rho), ':.', label = 'Messwerte')
plt.semilogy(t_plot_rho, N(t_plot_rho, *params_rho) , label = 'Regressionkurve')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$N/s^{-1}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()

plt.semilogy(t_van, np.log(imp_van), ':.', label = 'Messwerte')
plt.semilogy(t_plot_van, N(t_plot_van, *params_van), label = 'Regressionkurve')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$N/s^{-1}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()
