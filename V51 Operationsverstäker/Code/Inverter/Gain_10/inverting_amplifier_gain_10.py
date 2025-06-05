import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read data from files
amp_1 = np.loadtxt('amp_1.txt')
amp_2 = np.loadtxt('amp_2.txt')

# Calculate U_gain
U_gain = amp_2 / amp_1
U_gain_used = U_gain[6:14]
U_gain_rest = U_gain[0:5]

print(U_gain_rest)

# Create subset of x_array for corresponding U_gain
x_array = np.array([1, 10, 20, 30, 40, 50, 70, 80, 90, 100, 200, 300, 500, 800])
x_array_used = x_array[6:14]
x_array_rest = x_array[0:5]

# Define regression function
def reg(f, a, b):
    return a * f**b

# Perform curve fitting
params, covariance_matrix = curve_fit(reg, x_array_used, U_gain_used)

# Berechne die Grenzfrequenz
v_const = np.mean(U_gain[0:5])
v_grenz = (v_const/(np.sqrt(2)*params[0]))**(1/params[1])

print(params)
print(np.sqrt(np.diag(covariance_matrix)))
print(v_const)

# Create data for the regression curve
temp_plot = np.linspace(20, 4000, 1000)

# Create a double logarithmic plot
plt.figure(figsize=(10, 6))
plt.loglog(x_array[6:14], U_gain[6:14], 'o', label="Messwerte", markersize=8)
plt.loglog(x_array[0:5], U_gain[0:5], 'x', label="Nicht genutzte Messwerte", color="grey", markersize=8)
plt.vlines(v_grenz, 0.1, 100, linestyles="dashed", label="Grenzfrequenz", linewidth=2)
plt.loglog(temp_plot, reg(temp_plot, *params), label="Potenz-Fit", linewidth=2)
plt.xlim(0.1, 10000)
plt.ylim(0.1, 100)

# Add labels, grid, and enhancements
plt.title("Frequenz vs Gain - Doppelt logarithmischer Plot", fontsize=16, pad=15)
plt.xlabel("Frequenz (Hz)", fontsize=12)
plt.ylabel("Gain", fontsize=12)
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Save as a PDF and display
plt.savefig("Prettier_Graph.pdf", format="pdf", dpi=300)
plt.show()