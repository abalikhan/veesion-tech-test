import os
import torch
import numpy as np
import cv2
import mediapipe as mp
from models.skeleton_lstm_model import SkeletonLSTMClassifier
from utils.skeleton_extraction import process_video
import argparse

# Load our model
def load_model(model_path, input_dim, num_classes, device):
    model = SkeletonLSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    if os.path.exists(model_path):  # Because it's trained on dummy data and acc is always zero so no model was saved during validation
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(model, sequence, input_size, seq_len, device):
    if seq_len:
        if len(sequence) >= seq_len:
            sequence = sequence[:seq_len]
        else:
            pad = np.zeros((seq_len - len(sequence), input_size))
            sequence = np.vstack([sequence, pad])
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, 258]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return pred, probs.squeeze().cpu().numpy()

def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.input_size, args.num_classes, DEVICE)
    
    if args.input_path.endswith(".npy"):  # could be .json as well but here we are using only .npy files
        print(f"[INFO] Loading keypoints from: {args.input_path}")
        sequence = np.load(args.input_path)
    else:
        print(f"[INFO] Extracting keypoints from video: {args.input_path}")
        sequence = process_video(args.input_path)  # process_video is the function we created to extract skeleton keypoints for training

    pred_class, probs = run_inference(model, sequence, args.input_size, args.seq_len, DEVICE)
    print(f"\n Predicted class: {pred_class} | Probabilities: {probs.round(3)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Skeleton LSTM Classifier")
    parser.add_argument('--input_path', type=str, default="./data/videos/S023C001P063R001A098_rgb.avi",
                        help="Input .npy skeleton file or raw video file")
    parser.add_argument('--model_path', type=str, default="best_model_task1.pth")
    parser.add_argument('--input_size', type=int, default=258)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=None, help="Truncate or pad sequence to this length (default: no truncation/pad)")

    args = parser.parse_args()
    main(args)
