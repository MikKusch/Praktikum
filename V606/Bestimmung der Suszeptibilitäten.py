import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties as unp


temp = 294

def lande(J, L, S):
    return (3*J*(J+1)+(S*(S+1)-L*(L+1)))/(2*J*(J+1))


def magMo(rho, M):
    return 2*(rho/M)*const.N_A

#               J,   L,  S
Dy = np.array([ 7.5, 5, 2.5])

Nd = np.array([ 4.5, 6, 1.5])

Gd = np.array([ 3.5, 0, 3.5])

print('Lande-Faktor von Dy2O3: ', lande(Dy[0], Dy[1], Dy[2]))

print('Lande-Faktor von Nd2O3: ', lande(Nd[0], Nd[1], Nd[2]))

print('Lande-Faktor von Gd2O3: ', lande(Gd[0], Gd[1], Gd[2]))


gd_theo = (const.mu_0*const.physical_constants['Bohr magneton'][0]**2*lande(Gd[0], Gd[1], Gd[2])**2*2.46*10**28*Gd[0]*(Gd[0]+1))/(3*const.k*294)

nd_theo = (const.mu_0*const.physical_constants['Bohr magneton'][0]**2*lande(Nd[0], Nd[1], Nd[2])**2*2.59*10**28*Nd[0]*(Nd[0]+1))/(3*const.k*294)

dy_theo = (const.mu_0*const.physical_constants['Bohr magneton'][0]**2*lande(Dy[0], Dy[1], Dy[2])**2*2.46*10**28*Dy[0]*(Dy[0]+1))/(3*const.k*294)

print(gd_theo)
print(nd_theo)
print(dy_theo)

