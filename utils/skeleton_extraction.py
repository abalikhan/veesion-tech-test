import cv2
import mediapipe as mp
import os
import numpy as np

# Skeleton keypoints extraction function
def extract_keypoints(results):
    keypoints = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0] * (33 * 4))  # 33 pose keypoints

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * (21 * 3))  # 21 hand keypoints

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * (21 * 3))

    return keypoints


# Process a single video and return keypoints array
def process_video(video_path):
    mp_holistic = mp.solutions.holistic
    all_keypoints = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            keypoints = extract_keypoints(results)
            all_keypoints.append(keypoints)
        cap.release()

    return np.array(all_keypoints)


if __name__ == '__main__':
    VIDEO_DIR = "./data/videos"
    OUTPUT_DIR = "./data/keypoints"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')] # for this code we are using only .avi, but they can be changed to accomodation other video types

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"[INFO] Processing: {video_file}")
        keypoints_array = process_video(video_path)

        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(output_path, keypoints_array)
        print('Keypoints saved successfully ...!')
