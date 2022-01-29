import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

freq, rad, a = np.genfromtxt('Daten3.txt', unpack=True)

plt.semilogy(freq, rad, '.', label='Messwerte')

plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\phi/rad$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_4.pdf')
plt.show()


plt.plot(freq[8:15:], rad[8:15:], ':.', label='Messwerte')
plt.xlabel(r'$f/kHz$')
plt.ylabel(r'$\phi/rad$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_5.pdf')


