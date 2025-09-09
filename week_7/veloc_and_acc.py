import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import re

# === Parameters ===
dataset_dir = r"/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision"
csv_labels = r"/Users/simeonstamboliyski/Desktop/GATE/video_info.csv"
results_dir = r"/Users/simeonstamboliyski/Desktop/GATE/week_7/plots"
fv_name = r"/Users/simeonstamboliyski/Desktop/GATE/week_7/fv_le2i_SF_Tv_max.mat"

N = 190        # total number of videos
T = 1000        # max number of frames per video (padded if shorter)
dt = 1 / 25.0  # frame interval
sigma = 4      # Gaussian smoothing

os.makedirs(results_dir, exist_ok=True)

# Load CSV
labels_df = pd.read_csv(csv_labels)

# Helper: extract environment folder from full path
def get_env_folder(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))

# Helper: extract the video number in parentheses
def get_video_number(path):
    match = re.search(r'\((\d+)\)', path)
    return int(match.group(1)) if match else 0

# Add columns for sorting
labels_df['env_folder'] = labels_df['video_path'].apply(get_env_folder)
labels_df['video_num'] = labels_df['video_path'].apply(get_video_number)

# Sort first by environment, then by video number
labels_df = labels_df.sort_values(by=['env_folder', 'video_num']).reset_index(drop=True)

# Extract fall labels
fall_values = labels_df["fall_label"].tolist()

# === Initialize arrays ===
vyt = np.zeros((N, T))

# === Helper: compute vertical velocity ===
def compute_vertical_velocity(video_path, max_frames=T):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.zeros(max_frames)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    vy = []
    while True:
        ret, frame = cap.read()
        if not ret or len(vy) >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        # vertical component
        vy.append(np.mean(flow[..., 1]))
        prev_gray = gray
    cap.release()
    # pad if shorter than max_frames
    if len(vy) < max_frames:
        vy.extend([0] * (max_frames - len(vy)))
    return np.array(vy[:max_frames])

# === Step 1: compute vertical velocities for all videos ===
for i, video_path in enumerate(labels_df["video_path"]):
    vyt[i, :] = compute_vertical_velocity(video_path)
    print(f"Processing video: {video_path}")

# === Step 2: smooth velocities and compute accelerations ===
vs = gaussian_filter1d(vyt, sigma=sigma, axis=1)
at = np.diff(vs, axis=1) / dt
at = np.pad(at, ((0, 0), (0, 1)), 'constant')  # pad to match shape

# === Step 3: extract features ===
amax = np.max(at, axis=1)
amin = np.min(at, axis=1)
amaxt = np.argmax(at, axis=1)
amint = np.argmin(at, axis=1)

vmax = np.max(vs, axis=1)
vmin = np.min(vs, axis=1)
vmaxt = np.argmax(vs, axis=1)
vmint = np.argmin(vs, axis=1)

# === Step 4: build feature vector ===
fvsize = 4
feature_vector = np.zeros((N, fvsize))
for i in range(N):
    feature_vector[i, :] = [vmax[i], amax[i], amin[i], fall_values[i]]

# === Step 5: save plots ===
for i in range(N):
    plt.figure(figsize=(10, 5))
    plt.plot(vyt[i, :], label="Raw velocity", alpha=0.6)
    plt.plot(vs[i, :], label="Smoothed velocity", linewidth=2)
    plt.plot(at[i, :], label="Acceleration", linewidth=1.5)
    plt.title(f"Video {i+1} | Fall: {fall_values[i]}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"video_{i+1}.png"))
    plt.close()

# === Step 6: save feature vector ===
import scipy.io as sio
os.makedirs(os.path.dirname(fv_name), exist_ok=True)
sio.savemat(fv_name, {"featureVector": feature_vector})

print("Processing complete. Features and plots saved.")