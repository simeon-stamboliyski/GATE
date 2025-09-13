from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "file1.mat")

data1 = loadmat(file_path)
signal = data1['y']

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
plt.title('EEG first file Green Channels')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()