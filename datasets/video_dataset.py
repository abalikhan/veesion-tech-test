# datasets/video_dataset.py

import os
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from utils.preprocess_video import preprocess_video

class VideoDataset(Dataset):
    """
    Video-to-frame dataset for our end-to-end modelling.

    Inputs a video

    Returns per item:
      pixel_values: Tensor[T, C, H, W]
      label: int
    """
    def __init__(
        self,
        video_dir: str,
        label_csv: str,
        processor_name: str = "facebook/dinov2-base",
        seq_len: int = None
    ):
        # video file paths
        self.video_paths = sorted(
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".avi", ".mp4"))
        )

        # read labels to dictionary
        label_df = pd.read_csv(label_csv)
        self.labels = {row['video']: row['label'] for _, row in label_df.iterrows()}

        # Build the DINOv2 image processor
        self.processor = AutoImageProcessor.from_pretrained(
            processor_name, trust_remote_code=True
        )

        # fixed sequence length (truncate or pad)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video and find its label
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label = self.labels.get(video_name, -1)

        # Extract and preprocess frames
        pixel_values = preprocess_video(video_path=video_path, processor=self.processor, seq_len=self.seq_len)

        return pixel_values, label
