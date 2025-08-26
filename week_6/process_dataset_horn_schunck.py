import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def horn_schunck(prev_gray, gray, alpha=1.0, iterations=100):
    """Compute optical flow using the Horn-Schunck method."""
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = gray.astype(float) - prev_gray.astype(float)

    u = np.zeros_like(prev_gray, dtype=float)
    v = np.zeros_like(prev_gray, dtype=float)

    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,  0,  1/6],
                       [1/12, 1/6, 1/12]])

    for _ in range(iterations):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        der = (Ix*u_avg + Iy*v_avg + It) / (alpha**2 + Ix**2 + Iy**2 + 1e-5)
        u = u_avg - Ix * der
        v = v_avg - Iy * der

    return u, v


def process_video(video_path, out_dir, alpha=1.0, iterations=50):
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

        flow_x, flow_y = horn_schunck(prev_gray, gray, alpha=alpha, iterations=iterations)

        flow_x_mins.append(np.min(flow_x))
        flow_x_maxs.append(np.max(flow_x))
        flow_x_avgs.append(np.mean(flow_x))

        flow_y_mins.append(np.min(flow_y))
        flow_y_maxs.append(np.max(flow_y))
        flow_y_avgs.append(np.mean(flow_y))

        del flow_x, flow_y

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
    plt.title("Optical Flow X (per frame)")
    plt.xlabel("Frame")
    plt.ylabel("Flow X")
    plt.legend()
    plt.grid(True)

    # Flow-Y subplot
    plt.subplot(2, 1, 2)
    plt.plot(frames, flow_y_mins, label="min", color='blue', linewidth=1.5)
    plt.plot(frames, flow_y_maxs, label="max", color='red', linewidth=1.5)
    plt.plot(frames, flow_y_avgs, label="avg", color='green', linewidth=1.5)
    plt.title("Optical Flow Y (per frame)")
    plt.xlabel("Frame")
    plt.ylabel("Flow Y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the figure
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_file = os.path.join(out_dir, f"{video_name}_flow.png")
    plt.savefig(out_file)
    plt.close()

    print(f"Saved plot to {out_file}")


def natural_sort_key(s):
    """Helper to sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def process_dataset(parent_folder, alpha=1.0, iterations=50):
    """Process every video in the LE2i dataset structure and save results."""
    # Create results folder on Desktop
    desktop = os.path.expanduser("~/Desktop")
    results_dir = os.path.join(desktop, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Traverse dataset
    for root, dirs, files in os.walk(parent_folder):
        dirs.sort(key=natural_sort_key)   # ensure subfolders sorted
        if os.path.basename(root) == "Videos":  # Only go inside Videos folders
            rel_path = os.path.relpath(root, parent_folder)  # relative path from dataset root
            out_dir = os.path.join(results_dir, rel_path)    # mirror structure in results
            os.makedirs(out_dir, exist_ok=True)

            for file in sorted(files, key=natural_sort_key):
                if file.lower().endswith((".avi", ".mp4", ".mov")):
                    video_path = os.path.join(root, file)
                    print(f"Processing: {video_path}")
                    process_video(video_path, out_dir, alpha, iterations)


# === Example usage ===
process_dataset(
    parent_folder="/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision/Coffee_room_01/..", 
    alpha=1.0, 
    iterations=50
)