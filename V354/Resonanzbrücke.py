import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

freq, u_c = np.genfromtxt('Daten2.txt', unpack=True)

u = 0.8

u_q = u_c / u

plt.semilogy(freq, u_q, ':.', label='Messwerte')

plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\frac{U_C}{U}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
