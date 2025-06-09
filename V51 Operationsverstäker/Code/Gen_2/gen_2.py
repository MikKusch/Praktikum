import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import signal


def exp_decay(t, y0, tau, a):
    return y0 + a * np.exp(-t / tau)


x_axis, amp_1, amp_2 = pd.read_csv('scope_high.csv', delimiter=',', usecols=(0, 1, 2)).to_numpy().T

peaks, _ = signal.find_peaks(amp_2, height=0.15)
peaks = peaks[:15]  # Take only first 8 peaks
t_peaks = x_axis[peaks]
amp_peaks = amp_2[peaks]


popt, pcov = curve_fit(exp_decay, t_peaks, amp_peaks, p0=[0.5, 0.02, 1], maxfev=1000)

print("Tau: ", popt[1]*1000)

t_fit = np.linspace(min(t_peaks), max(t_peaks), 1000)


plt.plot(x_axis[:400], amp_2[:400], label="Messwerte")
plt.scatter(x_axis[peaks], amp_2[peaks], color='red', marker='x', s=100, label='Peaks')
plt.plot(t_fit, exp_decay(t_fit, *popt), label=r'Fit: $y₀ + a * \exp( -t / \tau )$')
plt.xlabel("Zeit (s)")
plt.ylabel("Amplitude (V)")
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig("scope_high_plot.pdf", format="pdf", dpi=300)
plt.show()
