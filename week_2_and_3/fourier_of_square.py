import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import square

# Parameters
fs = 1000         # Sampling rate
T = 1             # Duration in seconds
t = np.linspace(0, T, fs, endpoint=False)

# --- Square Wave ---
square_wave = square(2 * np.pi * 5 * t)
spectrum_square = fft(square_wave)
freqs = fftfreq(len(t), 1/fs)

plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.plot(t, square_wave)
plt.title("Square Wave (5 Hz)")

plt.subplot(2,1,2)
plt.stem(freqs[:fs//2], np.abs(spectrum_square)[:fs//2])
plt.title("FFT of Square Wave")
plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()