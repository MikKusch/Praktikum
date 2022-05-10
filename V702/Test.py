import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit




Nun = np.array([190, np.sqrt(190)])
Nu30n = Nun/20
Nu15n = Nun/40

print('Untergrund: ', *Nun)
print('Untergrund (30s): ', *Nu30n)
print('Untergrund (15s): ', *Nu15n)


def N(t, N0, λ):
    return N0*np.exp(-λ*t)

imp_van = np.genfromtxt('Daten_Vana.txt', unpack = True)
imp_van = np.array([imp_van, np.sqrt(imp_van)])

sourceFile = open('N_gemessen.txt', "w")
print(*imp_van, file=sourceFile)
sourceFile.close()

n = imp_van[0] - Nu30n[0]
n_fe = imp_van[1] - Nu30n[1]

sourceFile = open('N_30s.txt', "w")
print(n, n_fe, file=sourceFile)
sourceFile.close()

imp_rho = np.genfromtxt('Daten_Rhod.txt', unpack = True)
imp_rho = np.array([imp_rho, np.sqrt(imp_rho)])

sourceFile = open('N_gemessen_2.txt', "a")
print(imp_rho[1], file=sourceFile)
sourceFile.close()

n_2 = imp_rho[0] - Nu15n[0]
n_fe_2 = imp_rho[1] - Nu15n[1]

sourceFile = open('N_15s.txt', "a")
print(n_2, n_fe_2, file=sourceFile)
sourceFile.close()

t_van = np.linspace(0, 900, 30)
t_rho = np.linspace(0, 750, 50)

params_van, covariance_matrix_van = curve_fit(N, t_van, n, p0 = (100,0.003))
uncertainties_van = np.sqrt(np.diag(covariance_matrix_van))

print('Parameter (Vana): ', *params_van)
print('Fehler (Vana): ', *uncertainties_van)
T=np.log(2)/params_van[1]
T_fe = uncertainties_van[1]/params_van[1]*100
T_fe = T_fe * T /100
print('Halbwertszeit: ', T, '+-', T_fe)

params_rho_k, covariance_matrix_rho_k = curve_fit(N, t_rho[:15:], n_2[:15:], p0 = (1000,0.003))
uncertainties_rho = np.sqrt(np.diag(covariance_matrix_rho_k))

print('Parameter (Rho_kurz): ', *params_rho_k)
print('Fehler (Rho_kurz): ', *uncertainties_rho)
T=np.log(2)/params_rho_k[1]
T_fe = uncertainties_rho[1]/params_rho_k[1]*100
T_fe = T_fe * T /100
print('Halbwertszeit: ', T, '+-', T_fe)

params_rho_l, covariance_matrix_rho_l = curve_fit(N, t_rho[15::], n_2[15::], p0 = (1000,0.003))
uncertainties_rho_l = np.sqrt(np.diag(covariance_matrix_rho_l))

print('Parameter (Rho_lang): ', *params_rho_l)
print('Fehler (Rho_lang): ', *uncertainties_rho_l)
T=np.log(2)/params_rho_l[1]
T_fe = uncertainties_rho_l[1]/params_rho_l[1]*100
T_fe = T_fe * T /100
print('Halbwertszeit: ', T, '+-', T_fe)


t_plot_van = np.linspace(0, 900, 901)
t_plot_rho = np.linspace(0, 750, 751)

plt.yscale('log')
plt.errorbar(t_van, n, yerr=n_fe, fmt='k.', label='Vanadium')
plt.plot(t_plot_van, N(t_plot_van, *params_van), label = 'Regressionkurve', color = 'green')


plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Vanadium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('Graph_2.pdf')
plt.clf()

plt.errorbar(t_rho, n_2, yerr=n_fe_2, fmt='k.', label='Rhodium')
plt.axvline(x=225, ymin=0, ymax=1, label = r'$f^*$', color = 'green')
plt.plot(t_plot_rho[:225:], N(t_plot_rho[:225:], *params_rho_k) , label = 'kurzer Zerfall')
plt.plot(t_plot_rho[225::], N(t_plot_rho[225::], *params_rho_l) , label = 'langer Zerfall')

plt.yscale('log')
plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Rhodium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('Graph_1.pdf')
plt.clf()


plt.errorbar(t_rho[:15:], n_2[:15:], yerr=n_fe_2[:15:], fmt='k.', label='Rhodium')
plt.plot(t_plot_rho[:225:], N(t_plot_rho[:225:], *params_rho_k) , label = 'kurzer Zerfall')

plt.yscale('log')
plt.xlabel(r'$t/s$')
plt.ylabel(r'$log(N)$')
plt.title('Rhodium')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('Graph_3.pdf')