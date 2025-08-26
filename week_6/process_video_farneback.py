import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_video_with_farneback(video_path):
    """Run Farneback optical flow on a video and plot flow_y stats."""
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
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x, flow_y = flow[...,0], flow[...,1]
        
        flow_y_mins.append(np.min(flow_y))
        flow_y_maxs.append(np.max(flow_y))
        
        # Visualization
        flow_vis = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_vis_rgb = cv2.cvtColor(flow_vis, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((flow_vis_rgb, frame))
        
        cv2.imshow('Farneback Flow (left) and Video (right)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()
    
    plt.figure(figsize=(10,5))
    plt.plot(flow_y_mins, label='Min flow_y')
    plt.plot(flow_y_maxs, label='Max flow_y')
    plt.title('Min and Max of flow_y per Frame (Farneback)')
    plt.xlabel('Frame Index')
    plt.ylabel('Flow Y Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Example usage ===
process_video_with_farneback('/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision/Coffee_room_01/Videos/video (9).avi')