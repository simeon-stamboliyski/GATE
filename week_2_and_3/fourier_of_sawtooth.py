import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import sawtooth

fs = 1000
T = 1             
t = np.linspace(0, T, fs, endpoint=False)

sawtooth_wave = sawtooth(2 * np.pi * 5 * t)
spectrum_saw = fft(sawtooth_wave)
freqs = fftfreq(len(t), 1/fs)

plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.plot(t, sawtooth_wave)
plt.title("Sawtooth Wave (5 Hz)")

plt.subplot(2,1,2)
plt.stem(freqs[:fs//2], np.abs(spectrum_saw)[:fs//2])
plt.title("FFT of Sawtooth Wave")
plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()