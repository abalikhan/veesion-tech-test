import cv2
import os

def extract_ssl_frames(video_dir, output_dir, frame_rate=5):
    """
    Extract one frame every `frame_rate` frames from each video in video_dir.
    Saves as JPGs in output_dir.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)
    videos = sorted(f for f in os.listdir(video_dir)
                    if f.lower().endswith('.avi'))
    for vid in videos:
        path = os.path.join(video_dir, vid)
        cap = cv2.VideoCapture(path)
        basename = os.path.splitext(vid)[0]
        count = saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_rate == 0:
                fname = f"{basename}_f{saved:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), frame)
                saved += 1
            count += 1

        cap.release()
        print(f"Extracted {saved} frames from {vid}")

# Run the function
if __name__ == "__main__":
    extract_ssl_frames(
        video_dir="./data/videos",
        output_dir="./data/frames",
        frame_rate=5
    )