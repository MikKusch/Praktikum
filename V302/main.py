import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



x, y = np.genfromtxt('Daten.txt', unpack = True)



x_plot = np.linspace(x[0], x[-1], 1000)
y_plot = (1/9)*(x_plot**2-1)**2/((1-x_plot**2)**2+9*x_plot**2)
y_plot = np.sqrt(y_plot)


plt.semilogx(x, y, ':.', label = 'Messwerte')
plt.semilogx(x_plot, y_plot, label = 'Theoriekurve')




plt.ylabel(r'$\frac{U_{Br}}{U_S}$')
plt.xlabel(r'$\Omega = \frac{\nu}{\nu_0}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Graph_1.pdf')