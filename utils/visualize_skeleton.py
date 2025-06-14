import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30),
    (31, 32), (27, 31), (28, 32)
]

def visualize_keypoints_matplotlib(video_path, keypoints_path, step=10):
    keypoints = np.load(keypoints_path)
    cap = cv2.VideoCapture(video_path)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= len(keypoints):
            break

        if idx % step == 0:
            kps = keypoints[idx]
            pose = np.array(kps[:33 * 4]).reshape(33, 4)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.axis('off')

            # Draw joints
            for i, (x, y, z, v) in enumerate(pose):
                if v > 0.5:
                    ax.scatter(x * w, y * h, c='lime', s=10)

            # Draw connections
            for i, j in POSE_CONNECTIONS:
                if pose[i][3] > 0.5 and pose[j][3] > 0.5:
                    x1, y1 = pose[i][0] * w, pose[i][1] * h
                    x2, y2 = pose[j][0] * w, pose[j][1] * h
                    ax.plot([x1, x2], [y1, y2], 'y-', linewidth=1)

            plt.title(f'Frame {idx}')
            plt.show()

        idx += 1

    cap.release()

if __name__ == "__main__":
    video_path = "./data/videos/S023C001P063R001A098_rgb.avi"
    keypoints_path = "./data/keypoints/S023C001P063R001A098_rgb.npy"
    visualize_keypoints_matplotlib(video_path, keypoints_path)
