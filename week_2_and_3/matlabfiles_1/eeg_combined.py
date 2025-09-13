import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# data1 = loadmat('file1.mat')
# data2 = loadmat('file2.mat')
# data3 = loadmat('file3.mat')

# signal1 = data1['y']
# print("Signal shape:", signal1.shape)
# print("Signal data type:", signal1.dtype)

# signal2 = data2['y']
# print("Signal shape:", signal2.shape)
# print("Signal data type:", signal2.dtype)

# signal3 = data3['y']
# print("Signal shape:", signal3.shape)
# print("Signal data type:", signal3.dtype)

# combined_signal = np.concatenate((signal1, signal2, signal3), axis=1)

# print("Combined signal shape:", combined_signal.shape)

# savemat('combined_signal.mat', {'y': combined_signal})

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "combined_signal.mat")

combined_data = loadmat(file_path)
signal = combined_data['y']

green_channels = [5, 8, 10, 12, 14, 16, 23, 30, 32, 34, 44, 48, 62]
green_signal = signal[green_channels, :]

num_samples = green_signal.shape[1]
fs = 250

duration_sec = num_samples / fs
print(f"Total recording length: {duration_sec:.2f} seconds")

t = np.arange(num_samples) / fs

plt.figure(figsize=(12, 8))
offset = 100

for i, ch_idx in enumerate(green_channels):
    plt.plot(t, green_signal[i] + i * offset, label=f'Channel {ch_idx}')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude + offset')
plt.title('EEG Green Channels')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# The code below checks the frequency spectrum for a 50hz spike

plt.figure(figsize=(12, 10))

for i, ch_idx in enumerate(green_channels):
    signal_fft = fft(green_signal[i])
    freqs = fftfreq(len(signal_fft), 1/fs)
    
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(signal_fft[:len(freqs)//2])
    
    plt.plot(positive_freqs, magnitude, label=f'Channel {ch_idx}')

plt.title("FFT of Green Channels")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100)
plt.grid(True)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# There is no peak whatsoever at the 50hz frequency bin

# Frequency ranges of common brain waves:
# Delta: 0.5 – 4 Hz
# Theta: 4 – 8 Hz
# Alpha: 8 – 13 Hz
# Beta: 13 – 30 Hz
# Gamma: 30 – 100 Hz (sometimes up to 50 Hz depending on source)

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Filter parameters for alpha waves
lowcut = 8.0
highcut = 13.0

# Filter each green channel
filtered_alpha = np.zeros_like(green_signal)
for i in range(green_signal.shape[0]):
    filtered_alpha[i, :] = bandpass_filter(green_signal[i, :], lowcut, highcut, fs)

plt.subplot(2, 1, 2)
for i, ch_idx in enumerate(green_channels):
    plt.plot(t, filtered_alpha[i, :] + i * offset, label=f'Channel {ch_idx}')
plt.title('Filtered Green Channels (Alpha Waves 8-13 Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude + offset')

plt.tight_layout()
plt.show()