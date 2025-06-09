import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read data from files
phase_10 = np.loadtxt('phase_10.txt')
phase_100 = np.loadtxt('phase_100.txt')
phase_150 = np.loadtxt('phase_150.txt')

# Create subset of x_array for corresponding U_gain
x_array = np.array([1, 10, 20, 30, 40, 50, 70, 80, 90, 100, 200, 300, 500, 800])
x_array_100 = np.array([1, 10, 20, 30, 40, 50, 70, 90, 100, 200])

# Create a double logarithmic plot
plt.figure(figsize=(10, 6))
plt.loglog(x_array / 1000, phase_10, 'o', label="Phase 10", markersize=8)
plt.loglog(x_array_100 / 1000, phase_100, 'o', label="Phase 100", markersize=8)
plt.loglog(x_array / 1000, phase_150, 'o', label="Phase 150", markersize=8)

# Add labels, grid, and enhancements
plt.xlabel("Frequenz (kHz)", fontsize=12)
plt.ylabel("Phase (Grad)", fontsize=12)
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Save as a PDF and display
plt.savefig("Phase_Graph.pdf", format="pdf", dpi=300)
plt.show()
