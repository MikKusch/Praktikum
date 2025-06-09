import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read data from files
amp_in,amp_out  = np.loadtxt('diff_amp.txt', delimiter =',', usecols =(0, 1), unpack = True)

# Calculate U_gain
U_gain = amp_out / amp_in

x_array = np.array([10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200])


def reg(f, a, b):
    return a * f ** b


# Perform curve fitting
params, covariance_matrix = curve_fit(reg, x_array, U_gain)

# Create data for the regression curve
temp_plot = np.linspace(10, 300, 1000)

print("Paramter: ", *params)
print("Fehler: ", *np.sqrt(np.diag(covariance_matrix)))

plt.loglog(x_array, U_gain, 'o', label="Messwerte", markersize=8)
plt.loglog(temp_plot, reg(temp_plot, *params), label="Potenz-Fit", linewidth=2)

plt.xlabel("Frequenz (Hz)")
plt.ylabel("Verstärkung")
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

# Save as a PDF and display
plt.savefig("Differentiator_Plot.pdf", format="pdf", dpi=300)
plt.show()