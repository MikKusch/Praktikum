import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

Nun = 190
Nu30n = Nun/20
Nu30s = np.std(Nun/20)
Nu30 = unp.uarray(Nu30n, Nu30s)
print('Nu30',Nu30)

imp_rho = np.genfromtxt('Daten_Rhod.txt', unpack = True)
imp_van = np.genfromtxt('Daten_Vana.txt', unpack = True)

imp_rho = imp_rho - Nu30n
imp_van = imp_van

t_rho = np.linspace(0, 750, 50)
t_van = np.linspace(0, 900, 30)


def N(t, b, λ):
    return λ*t+b



params_rho_k, covariance_matrix_rho_k = curve_fit(N, t_rho[:25:], np.log10(imp_rho[:25:]))
uncertainties_rho = np.sqrt(np.diag(covariance_matrix_rho_k))

params_rho_l, covariance_matrix_rho_l = curve_fit(N, t_rho[25::], np.log10(imp_rho[25::]))
uncertainties_rho_l = np.sqrt(np.diag(covariance_matrix_rho_l))

params_van, covariance_matrix_van = curve_fit(N, t_van, np.log10(imp_van))
uncertainties_van = np.sqrt(np.diag(covariance_matrix_van))


print('Rhodium (kurz): ', *params_rho_k)
print('Rhodium Fehler (kurz): ', *uncertainties_rho)
print('Rhodium (lang): ', *params_rho_l)
print('Vandaium: ', *params_van)


t_plot_rho = np.linspace(0, 750, 1000)
t_plot_van = np.linspace(0, 900, 1000)


#plt.plot(t_rho, np.log10(imp_rho), 'k.', label = 'Messwerte')
plt.errorbar(t_rho, np.log10(imp_rho), yerr=np.std(np.sqrt(np.log10(imp_rho)/7)), fmt='k.', label='Rhodium')
plt.axvline(x=375.5, ymin=0, ymax=1, label = r'$f^*$', color = 'green')
plt.plot(t_plot_rho[:500:], N(t_plot_rho[:500:], *params_rho_k) , label = 'kurzer Zerfall')
plt.plot(t_plot_rho[500::], N(t_plot_rho[500::], *params_rho_l) , label = 'langer Zerfall')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Rhodium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_1.pdf')
plt.show()
plt.clf()

plt.errorbar(t_van, np.log10(imp_van), yerr=np.std(np.sqrt(np.log10(imp_van)/7)), fmt='k.', label='Vanadium')
plt.plot(t_plot_van, N(t_plot_van, *params_van), label = 'Regressionkurve', color = 'green')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Vanadium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
plt.show()
