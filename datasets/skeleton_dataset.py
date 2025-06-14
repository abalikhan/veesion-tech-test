import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class SkeletonDataset(Dataset):
    def __init__(self, keypoints_dir, label_csv=None, max_frames=150, num_features=258):
        self.keypoints_dir = keypoints_dir
        self.max_frames = max_frames
        self.num_features = num_features

        self.file_list = sorted([f for f in os.listdir(keypoints_dir) if f.endswith('.npy')]) # can be restricted to .json file if we are using json


        label_df = pd.read_csv(label_csv)
        self.labels = {row['video']: row['label'] for _, row in label_df.iterrows()}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        video_name = os.path.splitext(file_name)[0]
        label = self.labels[video_name]

        keypoints = np.load(os.path.join(self.keypoints_dir, file_name))  # [T, 258]

        # Pad/trim to fixed length
        T = keypoints.shape[0]
        if T >= self.max_frames:
            keypoints = keypoints[:self.max_frames]
        else:
            pad = np.zeros((self.max_frames - T, self.num_features))
            keypoints = np.vstack([keypoints, pad])

        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
