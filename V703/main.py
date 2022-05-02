import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit


N, U, amp = np.genfromtxt('Daten_plateau.txt', unpack=True)
N=N/120

def n(U, m, b):
    return m*U+b

params, covariance_matrix = curve_fit(n, U[3:12:], N[3:12:])
uncertainties = np.sqrt(np.diag(covariance_matrix))

print('Parameter: ', *params)
print('Fehler: ', *uncertainties)

U_plot = np.linspace(380, 580, 1000)


plt.errorbar(U, N, yerr=np.std(N), fmt='k.', label='Messwerte')
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


z = Z(amp, N)/10**10
print(N)
print('Z: ', z)

plt.errorbar(amp, z, yerr=np.std(z), fmt='k.', label='Messwerte')

plt.xlabel(r' I/$\mu$ A')
plt.ylabel(r'Z/$10^{10}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
plt.show()



