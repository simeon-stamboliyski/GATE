import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
fs = 1000         # Sampling rate
T = 1             # Duration in seconds
f0 = 5            # Frequency of sine wave

t = np.linspace(0, T, fs, endpoint=False)
signal = np.sin(2 * np.pi * f0 * t)

# FFT
spectrum = fft(signal)
freqs = fftfreq(len(t), 1/fs)

# Plot
plt.subplot(2,1,1)
plt.plot(t, signal)
plt.title("Sine Wave (5 Hz)")

plt.subplot(2,1,2)
plt.stem(freqs[:fs//2], np.abs(spectrum)[:fs//2])
plt.title("Fourier Transform (Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()