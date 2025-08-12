from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data2 = loadmat('file2.mat')
signal = data2['y']

green_channels = [5, 8, 10, 12, 14, 16, 23, 30, 32, 34, 44, 48, 62]
green_signal = signal[green_channels, :]
num_samples = green_signal.shape[1]

fs = 250
t = np.arange(num_samples) / fs

plt.figure(figsize=(12, 8))
offset = 100

for i, ch_idx in enumerate(green_channels):
    plt.plot(t, green_signal[i] + i * offset, label=f'Channel {ch_idx}')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude + offset')
plt.title('EEG second file Green Channels')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()