'''
===============================================================================
Title:        mdct_lfilter.py
Description:  1-D MDCT using the SciPy lfilter operation
Author:       Sushmita Raj
Date:         09.01.2025
Reference:    https://github.com/TUIlmenauAMS/MRSP_Tutorials
===============================================================================
Chapter 13 - MDCT, Homework Problem 6a, MRSP book by Prof. Gerald Schuller, TU Ilmenau
Implementing MDCT using SciPy lfilter and testing for perfect reconstruction with some delay
Book: https://moodle.tu-ilmenau.de/pluginfile.php/76782/mod_resource/content/20/mainbook.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Input signal - ramp function
N = 32  # Length of the ramp
x_len = 3 * N # Length of the input signal
x = np.zeros(x_len)  # Input signal with zero-padding (to observe the phase delay)
x[:N] = np.arange(N)  # Ramp function

# Sine window / baseband / prototype filter
oFactor = 2 # increase for longer and better filters but more delay
winlen = oFactor * N # length of window
h_win = np.sin(np.pi/winlen * (np.arange(winlen)+0.5))

# Calculate the MDCT analysis filter coefficients
mdct_coeffs_ana = np.zeros((N,winlen))
for k in range(N):
    mdct_coeffs_ana[k, :] = -1 * h_win * np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(np.arange(winlen)+0.5+N/2))

# Analysis Filter
filtered_x = np.zeros((N,x_len))
for k in range(N):
    filtered_x[k] = lfilter(mdct_coeffs_ana[k], [1], x)

# Critical sampling
# Resample the signal - downsample and upsample by N - (only every Nth element remains)
resampled_x = np.zeros((N, x_len))
resampled_x[:, ::N] = filtered_x[:, ::N]

# Calculate the MDCT snythesis filter coefficients
mdct_coeffs_syn = np.zeros((N,winlen))
for k in range(N):
    mdct_coeffs_syn[k, :] = h_win * np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(np.arange(winlen)+0.5-N/2))

# Synthesis Filter
recon_x = np.zeros(x_len)
for k in range(N):
    recon_x += lfilter(mdct_coeffs_syn[k], [1], resampled_x[k])

# Plot the original ramp signal
plt.figure(figsize=(10, 6))
plt.plot(x, label="Original Signal", linewidth=2)
plt.title("Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot the original and filtered ramp
#plt.figure(figsize=(10, 5))
#plt.plot(ramp, label="Original Ramp", linestyle='--', linewidth=2)
#plt.plot(filtered_x, label="Filtered Ramp", linewidth=2)
#plt.title("Ramp Function Filtered by MDCT Coefficients")
#plt.xlabel("Sample Index")
#plt.ylabel("Amplitude")
#plt.legend()
#plt.grid()
#plt.show()

# Plot the reconstructed signal
plt.figure(figsize=(10, 6))
plt.plot(recon_x, label="Reconstructed Signal", linewidth=2)
plt.title("Reconstructed Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Original and Reconstructed Signals
plt.figure(figsize=(10, 6))
plt.plot(x, label="Original Signal")
plt.plot(recon_x, label="Reconstructed Signal", linestyle='--')
plt.title("MDCT Perfect Reconstruction Test (Ramp Signal)")
plt.legend()
plt.grid()
plt.show()
