import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

freq, a, b = np.genfromtxt('Daten3.txt', unpack=True)


phi=a/b*2*np.pi
print(phi)

plt.semilogx(freq, phi, '.', label='Messwerte')

plt.xlabel(r'$f/Hz$')
plt.ylabel(r'$\phi/rad$')
plt.xticks([10e3, 2*10e3, 3*10e3, 4*10e3, 5*10e3],
           [r"$10^{4}$", r"$2\cdot 10^{4}$", r"$3\cdot 10^{4}$", r"$4\cdot 10^{4}$", r"$5\cdot 10^{4}$"])

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_4.pdf')
plt.close()


plt.plot(freq[5:17:], phi[5:17:], '.', label='Messwerte')
plt.axhline(y=np.pi/4, xmin=0, xmax=1, label = r'$\nu_1$', ls = ':', c = 'mediumseagreen')
plt.axhline(y=np.pi*3/4, xmin=0, xmax=1, label = r'$\nu_2$', ls = ':', c = 'mediumseagreen')
plt.axhline(y=np.pi/2, xmin=0, xmax=1, label = r'$\nu_{res}$', ls = ':', c = 'red')
plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\phi/rad$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_5.pdf')


