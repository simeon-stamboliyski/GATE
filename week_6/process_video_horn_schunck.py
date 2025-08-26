import cv2
import numpy as np
import matplotlib.pyplot as plt

def horn_schunck(prev_gray, gray, alpha=1.0, iterations=100):
    """Compute optical flow using the Horn-Schunck method."""
    # Compute gradients
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = gray.astype(float) - prev_gray.astype(float)
    
    # Initialize flow
    u = np.zeros_like(prev_gray, dtype=float)
    v = np.zeros_like(prev_gray, dtype=float)
    
    # Averaging kernel for smoothness
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


def process_video_with_horn_schunck(video_path, alpha=1.0, iterations=50):
    """Run Horn-Schunck optical flow on a video and plot flow_y stats."""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    flow_y_mins, flow_y_maxs = [], []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute Horn-Schunck optical flow
        flow_x, flow_y = horn_schunck(prev_gray, gray, alpha=alpha, iterations=iterations)
        
        flow_y_mins.append(np.min(flow_y))
        flow_y_maxs.append(np.max(flow_y))
        
        # Visualization
        flow_vis = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_vis_rgb = cv2.cvtColor(flow_vis, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((flow_vis_rgb, frame))
        
        cv2.imshow('Horn-Schunck Flow (left) and Video (right)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()
    
    
    plt.figure(figsize=(10,5))
    plt.plot(flow_y_mins, label='Min flow_y')
    plt.plot(flow_y_maxs, label='Max flow_y')
    plt.title('Min and Max of flow_y per Frame (Horn-Schunck)')
    plt.xlabel('Frame Index')
    plt.ylabel('Flow Y Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Example usage ===
process_video_with_horn_schunck('/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision/Coffee_room_01/Videos/video (9).avi', alpha=1.0, iterations=50)