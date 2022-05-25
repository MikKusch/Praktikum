import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as sc
import scipy.constants as const

#a) Bragg Bedingung überprüfen
ang_b, N_b = np.genfromtxt('Braggbedingung.txt', unpack=True)
ang_b = ang_b/2


xmax = ang_b[np.argmax(N_b)]
ymax = N_b.max()
soll_b = 14


plt.axvline(x=xmax, label=r'$\theta_{B, exp}$', ls='--')
plt.axvline(x=soll_b, label=r'$\theta_{B, theo}$', color='red')
plt.plot(ang_b, N_b, 'k.', label='Messwerte')


plt.xlabel(r'$\theta_{GM} / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')

plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Bragg_Bed.pdf')
plt.show()
plt.clf()

abw = ((xmax/soll_b-1)*-1)*100

print('Maximum: ', xmax)
print('Abweichung vom Soll-Wert: ', abw)


#b) Analyse eines Emissionsspektrums

ang_em, N_em = np.genfromtxt('Emissionsspektrum.txt', unpack=True)
ang_em = ang_em/2

peaks = sc.find_peaks(N_em, height= 500)
height = peaks[1]['peak_heights']

peak_pos = ang_em[peaks[0]]

pos = peaks[0]

peak_wi = sc.peak_widths(N_em, pos, rel_height=0.5)

height_b = peak_wi[1]



left_b = peak_wi[2]
left_b = np.around(left_b)
left_b = left_b.astype(int)
b_min_1, b_min_2 = ang_em[left_b]

right_b = peak_wi[3]
right_b = np.around(right_b)
right_b = right_b.astype(int)
b_max_1, b_max_2 = ang_em[right_b]

x_l = (ang_em[left_b]-10)/20

x_r = (ang_em[right_b]-10)/20

width = ang_em[right_b]-ang_em[left_b]


height_ka = height_b[0]/height[1]*1.16
height_kb = height_b[1]/height[1]

plt.plot(ang_em[60::], N_em[60::], 'k.', label='Messwerte')
plt.scatter(peak_pos[0], height[0], label=r'$K_{\beta}$')
plt.scatter(peak_pos[1], height[1], label=r'$K_{\alpha}$')

print('K_{a,b} Theta: ', *peak_pos)
print('Fehler: ', 0.1)
print('Linker Anfang: ', b_min_1, b_min_2)
print('Rechter Anfang: ', b_max_1, b_max_2)
print('Höhe der Halbwertsbreite', *height_b)

def welle(theta):
    return 2 * (201.4*10**(-12))*np.sin(np.deg2rad(theta))

def energie(lambd):
    return (const.Planck*const.speed_of_light)/lambd

E_Kx = energie(welle(peak_pos))/const.elementary_charge
print('Energie der Kanten: ', *E_Kx)
E_dKa = energie(welle(b_min_1))/const.elementary_charge-energie(welle(b_max_1))/const.elementary_charge
print('Energie der Halbwertsbreiten K_a', E_dKa)
E_dKb = energie(welle(b_min_2))/const.elementary_charge-energie(welle(b_max_2))/const.elementary_charge
print('Energie der Halbwertsbreiten K_b', E_dKb)
E_dKx = np.array([E_dKa, E_dKb])
Af = E_Kx/E_dKx
print('Das Auflösungsvermögen ist: ', *Af)

#Abschirmungskonstanten#
E_Kabs = 8987.96
ryd_un, _, __ = const.physical_constants['Rydberg constant times hc in eV']


sig_1 = 29 - np.sqrt(E_Kabs/ryd_un)

sig_2 = 29 - (2*np.sqrt((29-sig_1)**2-((E_Kx[1])/ryd_un)))

sig_3 = 29 - (3*np.sqrt((29-sig_1)**2-((E_Kx[0])/ryd_un)))

print('Abschirmkonstante sig_1', sig_1)
print('Abschirmkonstante sig_2', sig_2)
print('Abschirmkonstante sig_3', sig_3)



plt.hlines(y=height_b[0], xmin= b_min_1, xmax=b_max_1, color='g', linestyle='-', label='Halbwertsbreite')
plt.vlines(x=ang_em[left_b[0]], ymin= 0, ymax= height_b[0], color='g')
plt.vlines(x=ang_em[right_b[0]], ymin= 0, ymax= height_b[0], color='g')

plt.hlines(y=height_b[1], xmin=b_min_2, xmax=b_max_2, color='b', linestyle='-', label='Halbwertsbreite')
plt.vlines(x=ang_em[left_b[1]], ymin= 0, ymax= height_b[1], color='b')
plt.vlines(x=ang_em[right_b[1]], ymin= 0, ymax= height_b[1], color='b')

plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Emission.pdf')
plt.show()
plt.clf()

#c) Analyse der Absorbtionsspektren

zi = 30
ger = 32
str = 38
gal = 31
zir = 40

def n(U, m, b):
    return m*U+b

def rvsline(Ik, m, n):
    return (Ik - n ) / m

def sigma(z, E_k):
    return z - np.sqrt(E_k/ryd_un - (const.alpha**2 * z**4)/4)



print('Zink: ')
ang_zi, N_zi = np.genfromtxt('Zink.txt', unpack=True)

min_zi = np.amin(N_zi)
max_zi = np.amax(N_zi)

x_zi = np.array([ang_zi[5], ang_zi[6]])
y_zi = np.array([N_zi[5], N_zi[6]])
zi_plot = np.linspace(ang_zi[5], ang_zi[6], 1000)

params_zi, covariance_matrix_zis = curve_fit(n, x_zi, y_zi)

print('Parameter: ', *params_zi)

Ik_zi = (min_zi+max_zi)/2
print('I_min: ', min_zi)
print('I_max: ', max_zi)
print('I_K', Ik_zi)

deg = rvsline(Ik_zi, *params_zi)
print('Winkel: ', deg)
print('Energie: ', energie(welle(deg))/const.elementary_charge)
print('Abschirmkonstante: ', sigma(30, energie(welle(deg))/const.elementary_charge))
E_zi = energie(welle(deg))/const.elementary_charge
print('Absobtionsenergie', E_zi)

plt.plot(zi_plot, n(zi_plot, *params_zi), label = 'Lineare Regression')
plt.plot(ang_zi, N_zi, 'k.', label='Messwerte')
plt.scatter([rvsline(Ik_zi, *params_zi)], [Ik_zi] ,s=40, marker='x', color='red', label=r'$\theta$')
plt.scatter([18.2], [min_zi], s=40, marker='x', color='orange', label=r'$I_{min}$')
plt.scatter([19.0], [max_zi], s=40, marker='x', color='blue', label=r'$I_{max}$')
plt.axhline(y=Ik_zi, xmin=0, xmax=1, color='g', linestyle='-', label=r'$I_K$')
plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Zink.pdf')
plt.show()
plt.clf()


print('Strontium: ')
ang_str, N_str = np.genfromtxt('Strontium.txt', unpack=True)

min_str = np.amin(N_str)
max_str = np.amax(N_str)

x_str = np.array([ang_str[12], ang_str[14]])
y_str = np.array([N_str[12], N_str[14]])
str_plot = np.linspace(ang_str[12], ang_str[14], 1000)

params_str, covariance_matrix_str = curve_fit(n, x_str, y_str)

print('Parameter: ', *params_str)

Ik_str = (min_str+max_str)/2
print('I_min: ', min_str)
print('I_max: ', max_str)
print('I_K', Ik_str)

deg = rvsline(Ik_str, *params_str)
print('Winkel: ', deg)
print('Energie: ', energie(welle(deg))/const.elementary_charge)
print('Abschirmkonstante: ', sigma(str, energie(welle(deg))/const.elementary_charge))
E_str =  energie(welle(deg))/const.elementary_charge
print('Absobtionsenergie', E_str)

plt.plot(str_plot, n(str_plot, *params_str), label='Lineare Regression')
plt.plot(ang_str, N_str, 'k.', label='Messwerte')
plt.scatter([rvsline(Ik_str, *params_str)], [Ik_str], s=40, marker='x', color='red', label=r'$\theta$')
plt.scatter([10], [min_str], s=40, marker='x', color='orange', label=r'$I_{min}$')
plt.scatter([11.5], [max_str], s=40, marker='x', color='blue', label=r'$I_{max}$')
plt.axhline(y=Ik_str, xmin=0, xmax=1, color='g', linestyle='-', label=r'$I_K$')
plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Strontium.pdf')
plt.show()
plt.clf()


print('Zirkonium: ')
ang_zir, N_zir = np.genfromtxt('Zirkonium.txt', unpack=True)

min_zir = np.amin(N_zir)
max_zir = np.amax(N_zir)

x_zir = np.array([ang_zir[6], ang_zir[7]])
y_zir = np.array([N_zir[6], N_zir[7]])
zir_plot = np.linspace(ang_zir[6], ang_zir[7], 1000)

params_zir, covariance_matrix_zir = curve_fit(n, x_zir, y_zir)

print('Parameter: ', *params_zir)

Ik_zir = (min_zir+max_zir)/2
print('I_min: ', min_zir)
print('I_max: ', max_zir)
print('I_K', Ik_zir)

deg = rvsline(Ik_zir, *params_zir)
print('Winkel: ', deg)
print('Energie: ', energie(welle(deg))/const.elementary_charge)
print('Abschirmkonstante: ', sigma(zir, energie(welle(deg))/const.elementary_charge))
E_zir =  energie(welle(deg))/const.elementary_charge
print('Absobtionsenergie', E_zir)

plt.plot(zir_plot, n(zir_plot, *params_zir), label='Lineare Regression')
plt.plot(ang_zir, N_zir, 'k.', label='Messwerte')
plt.scatter([rvsline(Ik_zir, *params_zir)], [Ik_zir], s=40, marker='x', color='red', label=r'$\theta$')
plt.scatter([9.5], [min_zir], s=40, marker='x', color='orange', label=r'$I_{min}$')
plt.scatter([10.7], [max_zir], s=40, marker='x', color='blue', label=r'$I_{max}$')
plt.axhline(y=Ik_zir, xmin=0, xmax=1, color='g', linestyle='-', label=r'$I_K$')
plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Zirkonium.pdf')
plt.show()
plt.clf()



print('Gallium: ')
ang_gal, N_gal = np.genfromtxt('Gallium.txt', unpack=True)

min_gal = np.amin(N_gal)
max_gal = np.amax(N_gal)

x_gal = np.array([ang_gal[4], ang_gal[6]])
y_gal = np.array([N_gal[4], N_gal[6]])
gal_plot = np.linspace(ang_gal[4], ang_gal[6], 1000)

params_gal, covariance_matrix_gal = curve_fit(n, x_gal, y_gal)

print('Parameter: ', *params_gal)

Ik_gal = (min_gal+max_gal)/2
print('I_min: ', min_gal)
print('I_max: ', max_gal)
print('I_K', Ik_gal)

deg = rvsline(Ik_gal, *params_gal)
print('Winkel: ', deg)
print('Energie: ', energie(welle(deg))/const.elementary_charge)
print('Abschirmkonstante: ', sigma(gal, energie(welle(deg))/const.elementary_charge))
E_gal =  energie(welle(deg))/const.elementary_charge
print('Absobtionsenergie', E_gal)

plt.plot(gal_plot, n(gal_plot, *params_gal), label='Lineare Regression')
plt.plot(ang_gal, N_gal, 'k.', label='Messwerte')
plt.scatter([rvsline(Ik_gal, *params_gal)], [Ik_gal], s=40, marker='x', color='red', label=r'$\theta$')
plt.scatter([17.0], [min_gal], s=40, marker='x', color='orange', label=r'$I_{min}$')
plt.scatter([17.8], [max_gal], s=40, marker='x', color='blue', label=r'$I_{max}$')
plt.axhline(y=Ik_gal, xmin=0, xmax=1, color='g', linestyle='-', label=r'$I_K$')
plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Gallium.pdf')
plt.show()
plt.clf()



print('Germanium: ')
ang_ger, N_ger = np.genfromtxt('Germanium.txt', unpack=True)

min_ger = np.amin(N_ger)
max_ger = np.amax(N_ger)

x_ger = np.array([ang_ger[4], ang_ger[6]])
y_ger = np.array([N_ger[4], N_ger[6]])
ger_plot = np.linspace(ang_ger[4], ang_ger[6], 1000)

params_ger, covariance_matrix_ger = curve_fit(n, x_ger, y_ger)

print('Parameter: ', *params_ger)

Ik_ger = (min_ger+max_ger)/2
print('I_min: ', min_ger)
print('I_max: ', max_ger)
print('I_K', Ik_ger)

deg = rvsline(Ik_ger, *params_ger)
print('Winkel: ', deg)
print('Energie: ', energie(welle(deg))/const.elementary_charge)
print('Abschirmkonstante: ', sigma(ger, energie(welle(deg))/const.elementary_charge))
E_ger =  energie(welle(deg))/const.elementary_charge
print('Absobtionsenergie', E_ger)

plt.plot(ger_plot, n(ger_plot, *params_ger), label='Lineare Regression')
plt.plot(ang_ger, N_ger, 'k.', label='Messwerte')
plt.scatter([rvsline(Ik_ger, *params_ger)], [Ik_ger], s=40, marker='x', color='red', label=r'$\theta$')
plt.scatter([15.8], [min_ger], s=40, marker='x', color='orange', label=r'$I_{min}$')
plt.scatter([16.6], [max_ger], s=40, marker='x', color='blue', label=r'$I_{max}$')
plt.axhline(y=Ik_ger, xmin=0, xmax=1, color='g', linestyle='-', label=r'$I_K$')
plt.xlabel(r'$\theta / $' '\u00b0')
plt.ylabel(r'N / (Imp/s)')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Germanium.pdf')
plt.show()
plt.clf()




E_ry = np.array([E_zi,E_str,E_zir,E_gal,E_ger])
E_ry = np.sqrt(E_ry)
Z_ry = np.array([zi, str, zir, gal, ger])

z_plot = np.linspace(30, 40, 1000)

params_ryd, covariance_matrix_ryd = curve_fit(n, Z_ry, E_ry)

print('Parameter: ', *params_ryd)
print('Rydberenergie: ', params_ryd[0]**2)
print('Rydbergfrequenz: ', (params_ryd[0]**2 *const.elementary_charge)/(const.Planck*10**(15)))

plt.plot(z_plot, n(z_plot, *params_ryd), label='Lineare Regression')
plt.plot(Z_ry, E_ry , 'k.', label='errechnete Punkte')
plt.ylabel(r'$\sqrt{E_K} / eV^{1/2}$')
plt.xlabel(r'Z')
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('rydberg.pdf')
plt.show()










