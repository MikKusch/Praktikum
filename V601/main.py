import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties as unp



U_a_k, I_a_k = np.genfromtxt('Daten_Kalt.txt', unpack=True)
U_a_w, I_a_w = np.genfromtxt('Daten_Warm.txt', unpack=True)



plt.plot(U_a_k, I_a_k, '.', label='25,9$^\circ$C')
plt.plot(U_a_w, I_a_w, '.', label='153,4$^\circ$C')
plt.yticks([])
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A / V$')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Plot1.pdf')
plt.show()
plt.clf()


k = 0

y = np.zeros(len(U_a_k) -1 )
x = np.zeros(len(U_a_k) -1 )


while(k < len(U_a_k) - 1):
    y[k] = abs((I_a_k[k+1] - I_a_k[k])/ (U_a_k[k+1] - U_a_k[k]))
    x[k] = U_a_k[k] + (U_a_k[k+1]-U_a_k[k])/2
    k += 1


j = 0

z = np.zeros(len(U_a_w) -1 )
w = np.zeros(len(U_a_w) -1 )

while(j < len(U_a_w) - 1):
    z[j] = abs((I_a_w[j+1] - I_a_w[j])/ (U_a_w[j+1] - U_a_w[j]))
    w[j] = U_a_w[j] + (U_a_w[j+1]-U_a_w[j])/2
    j += 1

max_Ua_k = np.amax(y)
print(max_Ua_k)


plt.plot(x, y, '.', label = '25,9$^\circ$C')
plt.plot(w, z, '.', label = '153,4$^\circ$C')
plt.axvline(x=8.67, linestyle = '--', c = 'grey')
plt.yticks([])
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A / V$')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Energieverteil.pdf')
plt.show()


