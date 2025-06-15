import os
import cv2
from PIL import Image

def preprocess_video(video_path, processor, seq_len):
    """
    Load video frames, convert to PIL, truncate/pad, and preprocess via HuggingFace image processor.
    Returns Tensor [T, C, H, W].
    """
    # Ensure the path is a file
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"No such video file: {video_path}")

    # Capture frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()

    # Fail fast if nothing was read
    if len(frames) == 0:
        raise RuntimeError(f"Could not decode any frames from {video_path}")

    # Truncate or pad (using the first frame as pad value)
    if seq_len is not None:
        if len(frames) >= seq_len:
            frames = frames[:seq_len]
        else:
            pad_frame = frames[0]
            frames += [pad_frame] * (seq_len - len(frames))

    # Preprocess all frames at once
    processed = processor(frames, return_tensors="pt")
    return processed["pixel_values"]  # shape [T, C, H, W]
