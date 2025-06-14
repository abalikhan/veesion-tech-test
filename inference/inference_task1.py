import os
import torch
import numpy as np
import cv2
import mediapipe as mp
from models.skeleton_lstm_model import SkeletonLSTMClassifier
from utils.skeleton_extraction import process_video

# Configuration
INPUT_PATH = "./data/videos/S023C001P063R001A098_rgb.avi"      # Can be .npy or video
MODEL_PATH = "best_model_task1.pth"
INPUT_SIZE = 258
NUM_CLASSES = 5
SEQ_LEN = None  # Can truncate/pad sequences
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load our model
model = SkeletonLSTMClassifier(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def run_inference(sequence):
    if SEQ_LEN:
        if len(sequence) >= SEQ_LEN:
            sequence = sequence[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(sequence), INPUT_SIZE))
            sequence = np.vstack([sequence, pad])

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, 258]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return pred, probs.squeeze().cpu().numpy()

if __name__ == "__main__":
    if INPUT_PATH.endswith(".npy"): # could be .json as well but here we are using only .npy files
        print(f"[INFO] Loading keypoints from: {INPUT_PATH}")
        sequence = np.load(INPUT_PATH)
    else:
        print(f"[INFO] Extracting keypoints from video: {INPUT_PATH}")
        sequence = process_video(INPUT_PATH) # process_video is the function we created to extract skeleton keypoints for training

    pred_class, probs = run_inference(sequence)
    print(f"\n Predicted class: {pred_class} | Probabilities: {probs.round(3)}")
