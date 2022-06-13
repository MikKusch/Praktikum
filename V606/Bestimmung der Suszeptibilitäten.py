import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties as unp


temp = 294

def lande(J, L, S):
    return (3*J*(J+1)+(S*(S+1)-L*(L+1)))/(2*J*(J+1))


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

F = 0.866

Q_Nd = 0.0662
Q_Gd = 0.115
Q_Dy = 0.12

#Mittelwert von der diff Wiederstand

Nd_r = np.array([0.08, 0.095, 0.12])

Gd_r = np.array([0.765, 0.765, 0.785])

Dy_r = np.array([1.53, 1.56, 1.59])


#Mittelwert

Nd_r3 = np.array([3.4, 3.4, 3.405])

Gd_r3 = np.array([3.365, 3.575, 3.57])

Dy_r3 = np.array([3.38, 3.38, 3.39])


Nd_U = np.array([0, 0, 0.07])
Gd_U = np.array([0.1, 0.12, 0.15])
Dy_U = np.array([0.05, 0.2, 0.34])


def diff_r(x):
    return np.mean(x)


def sus_r(diff, R, F, Q):
    return 2*(diff/R)*(F/Q)


def sus_u(F, Q, U_br):
    return 4*(F/Q)*(U_br/1000)

Nd_U = diff_r(Nd_U)
Gd_U = diff_r(Gd_U)
Dy_U = diff_r(Dy_U)

diff_Nd = diff_r(Nd_r)
diff_Gd = diff_r(Gd_r)
diff_Dy = diff_r(Dy_r)

R3 = 1000

print(diff_Nd, diff_Gd, diff_Dy)


print('Susität von Nd: ', sus_r(diff_Nd, R3, F, Q_Nd))
print('Susität von Gd: ', sus_r(diff_Gd, R3, F, Q_Gd))
print('Susität von Dy: ', sus_r(diff_Dy, R3, F, Q_Dy))


print('Susität von Nd: ', sus_u(F, Q_Nd, Nd_U))
print('Susität von Gd: ', sus_u(F, Q_Gd, Gd_U))
print('Susität von Dy: ', sus_u(F, Q_Dy, Dy_U))





