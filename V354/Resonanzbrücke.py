import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

freq, u_c = np.genfromtxt('Daten2.txt', unpack=True)
u = 0.8

u_q = u_c / u
c = 1/np.sqrt(2)
plt.semilogy(freq, u_q, ':.', label='Messwerte')



plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\frac{U_C}{U}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_2.pdf')
plt.show()


plt.semilogy(freq[8:15:], u_q[8:15:], ':.', label='Messwerte')
plt.axvline(x=32.5, ymin=0, ymax=1, label = 'Resonanzüberhöhung', ls = ':', c = 'red')
plt.axhline(y=c, xmin=0, xmax=1, label = 'Halbwertsbreite', ls = ':')
plt.axvline(x=27, ymin=0, ymax=1, label = r'$\nu_-$', ls = ':', c = 'tab:olive')
plt.axvline(x=38.5, ymin=0, ymax=1, label = r'$\nu_+$', ls = ':', c = 'mediumseagreen')

plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\frac{U_C}{U}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_3.pdf')
plt.show()

