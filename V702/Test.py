import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit




Nun = np.array([190, np.sqrt(190)])
Nu30n = Nun/20

def N(t, b, λ):
    return λ*t+b

imp_van = np.genfromtxt('Daten_Vana.txt', unpack = True)
imp_van = np.array([imp_van, np.sqrt(imp_van)])

n = imp_van[0] - Nu30n[0]
n_fe = imp_van[1] - Nu30n[1]

t_van = np.linspace(0, 900, 30)

params_van, covariance_matrix_van = curve_fit(N, t_van, np.log10(n))
uncertainties_van = np.sqrt(np.diag(covariance_matrix_van))


t_plot_van = np.linspace(0, 900, 1000)


plt.errorbar(t_van, np.log10(n), yerr=np.log10(n_fe), fmt='k.', label='Vanadium')
plt.plot(t_plot_van, N(t_plot_van, *params_van), label = 'Regressionkurve', color = 'green')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Vanadium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()
