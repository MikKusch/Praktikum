import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as sc
import scipy.constants as const


U_a1, I_a1 = np.genfromtxt('Kurve_180Grad.txt', unpack=True)

peaks = sc.find_peaks(I_a1, height = 5, distance=10)
height = peaks[1]['peak_heights']

peak_pos = U_a1[peaks[0]]
pos = peaks[0]



diff_1 = peak_pos[1]-peak_pos[0]
diff_2 = peak_pos[2]-peak_pos[1]
diff_3 = peak_pos[4]-peak_pos[2]
diff_4 = peak_pos[6]-peak_pos[4]

diff = np.array([diff_1, diff_2, diff_3, diff_4])
diff1 = np.mean(diff)

print(peak_pos)
print(diff1)
print(diff)

pos = peaks[0]

plt.plot(U_a1, I_a1, 'k.', label='Messwerte')
plt.scatter(peak_pos[0], height[0], label=r'Maxima', c='red')
plt.scatter(peak_pos[1], height[1], c='red')
plt.scatter(peak_pos[2], height[2], c='red')
plt.scatter(peak_pos[4], height[4], c='red')
plt.scatter(peak_pos[6], height[6], c='red')


plt.axvline(x=peak_pos[0], linestyle = '--', c = 'grey')
plt.axvline(x=peak_pos[1], linestyle = '--', c = 'grey')
plt.axvline(x=peak_pos[2], linestyle = '--', c = 'grey')
plt.axvline(x=peak_pos[4], linestyle = '--', c = 'grey')
plt.axvline(x=peak_pos[6], linestyle = '--', c = 'grey')
plt.ylabel(r'$I_A$')
plt.xlabel(r'$U_A / V$')



plt.yticks([])
plt.legend()
plt.savefig('Frank180.pdf')
plt.show()
plt.clf()
