import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties as unp



frq, Ua, Ua_fe = np.genfromtxt('Filterkurve.txt', unpack=True)


def lorentz(x, x_0, a, gamma):
    return a/((x**2-x_0**2)**2+(gamma**2 * x_0**2))

def rvsline(V, x_0, a, gamma):
    return np.sqrt(np.sqrt(a/V-gamma**2 * x_0**2)+x_0**2)


max_Ua = np.amax(Ua)
max_Ua *= 1/np.sqrt(2)


params, covariance_matrix = curve_fit(lorentz, frq, Ua)

print('Durchlassfrequenz: ', *params)


frq_plot = np.linspace(frq[0], frq[-1], 1000)

diff = rvsline(max_Ua, *params)-params[0]
print('Nu_1', params[0]-diff)
print('Nu_2', params[0]+diff)
gu = params[0]/((params[0]+diff)-(params[0]-diff))
print('Güte', gu)



plt.plot(frq_plot, lorentz(frq_plot, *params), label = 'Lineare Regression')
plt.plot(frq, Ua, 'k.', label='Messwerte')
plt.hlines(y=max_Ua, xmin=20, xmax=40, color='g', linestyle='-', label=r'$\frac{U_{A, max}}{\sqrt{2}}$')
#plt.vlines(x=params[0]+diff, ymin= 0, ymax=4, color='g')
#plt.vlines(x=params[0]-diff, ymin= 0, ymax=4, color='g')
plt.xlabel(r'$\nu / kHz$')
plt.ylabel(r'$U_A / V$')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Filter.pdf')
plt.show()
plt.clf()






