import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


N, U, amp = np.genfromtxt('Daten_plateau.txt', unpack=True)
N = N/120
Nfe = np.sqrt(N*120)/120

def n(U, m, b):
    return m*U+b

params, covariance_matrix = curve_fit(n, U[3:12:], N[3:12:])
uncertainties = np.sqrt(np.diag(covariance_matrix))

print('Parameter: ', *params)
print('Fehler: ', *uncertainties)

U_plot = np.linspace(380, 580, 1000)


plt.errorbar(U, N, yerr=Nfe, fmt='k.', label='Messwerte')
plt.plot(U_plot, n(U_plot, *params) , label = 'Plateau')


plt.xlabel(r'Spannung U/V')
plt.ylabel(r'Impulsrate')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_1.pdf')
plt.show()


#b) Totzeit des Zählrohrs


N_1 = np.array([19535/120, ])
N_2 = np.array([17165/120, ])
N_3 = np.array([35999/120, ])


T = (N_1 + N_2-N_3)/(2 * N_1 * N_2)*10**6
t = np.sqrt(T)
print('Totzeit: ', *T, *t)

#c) Bestimmung der pro Teilchen vom Zählrohr freigesetzten Ladungen


def Z(I, N_0):
    return (I*10**(-6))/(sc.elementary_charge*N_0)

def f(x, m, b):
    return m*x+b

z = Z(amp, N)/10**10
print('Z: ', z)

amp_fe = 0.05/amp*100
N_fe = Nfe/N*100

zfe = amp_fe + N_fe
zfe = z*zfe/100

print(zfe)

para, cov_matr = curve_fit(f, amp, z)

amp_plot = np.linspace(amp[0], amp[-1], 1000)

plt.errorbar(amp, z, yerr=zfe, fmt='k.', label='Messwerte')
plt.plot(amp_plot, f(amp_plot, *para), label = 'Regressionkurve')

plt.xlabel(r' I/$\mu$ A')
plt.ylabel(r'Z/$10^{10}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
plt.show()



