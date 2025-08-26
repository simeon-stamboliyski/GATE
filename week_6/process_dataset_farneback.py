import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def process_video_farneback(video_path, out_dir):
    """Compute Farneback optical flow for a video and save min/max/avg plots."""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow_x_mins, flow_x_maxs, flow_x_avgs = [], [], []
    flow_y_mins, flow_y_maxs, flow_y_avgs = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # Collect min/max/avg stats
        flow_x_mins.append(np.min(flow_x))
        flow_x_maxs.append(np.max(flow_x))
        flow_x_avgs.append(np.mean(flow_x))

        flow_y_mins.append(np.min(flow_y))
        flow_y_maxs.append(np.max(flow_y))
        flow_y_avgs.append(np.mean(flow_y))

        del flow, flow_x, flow_y
        prev_gray = gray

    cap.release()

    # Plot results
    frames = np.arange(len(flow_x_mins))
    plt.figure(figsize=(10, 6))

    # Flow-X subplot
    plt.subplot(2, 1, 1)
    plt.plot(frames, flow_x_mins, label="min", color='blue', linewidth=1.5)
    plt.plot(frames, flow_x_maxs, label="max", color='red', linewidth=1.5)
    plt.plot(frames, flow_x_avgs, label="avg", color='green', linewidth=1.5)
    plt.title("Farneback Optical Flow X (per frame)")
    plt.xlabel("Frame")
    plt.ylabel("Flow X")
    plt.legend()
    plt.grid(True)

    # Flow-Y subplot
    plt.subplot(2, 1, 2)
    plt.plot(frames, flow_y_mins, label="min", color='blue', linewidth=1.5)
    plt.plot(frames, flow_y_maxs, label="max", color='red', linewidth=1.5)
    plt.plot(frames, flow_y_avgs, label="avg", color='green', linewidth=1.5)
    plt.title("Farneback Optical Flow Y (per frame)")
    plt.xlabel("Frame")
    plt.ylabel("Flow Y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_file = os.path.join(out_dir, f"{video_name}_flow_farneback.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot to {out_file}")


def natural_sort_key(s):
    """Helper to sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def process_dataset(parent_folder):
    """Process every video in the dataset structure and save plots."""
    desktop = os.path.expanduser("~/Desktop")
    results_dir = os.path.join(desktop, "results")
    os.makedirs(results_dir, exist_ok=True)

    for root, dirs, files in os.walk(parent_folder):
        dirs.sort(key=natural_sort_key)
        if os.path.basename(root) == "Videos":
            rel_path = os.path.relpath(root, parent_folder)
            out_dir = os.path.join(results_dir, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            for file in sorted(files, key=natural_sort_key):
                if file.lower().endswith((".avi", ".mp4", ".mov")):
                    video_path = os.path.join(root, file)
                    print(f"Processing: {video_path}")
                    process_video_farneback(video_path, out_dir)


# === Example usage ===
process_dataset(
    parent_folder="/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision/Coffee_room_01/.."
)