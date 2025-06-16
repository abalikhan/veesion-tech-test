# Veesion Technical Test – Abid Ali  
_Building a concise, end-to-end pipeline for temporal human gesture classification_

---

## 1.1 .Quick setup

```bash
# Create environment (example)
conda create -n veesion-env python=3.10 -y
conda activate veesion-env

# Install PyTorch with CUDA 11.8 (works on RTX 3000)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project requirements
pip install -r requirements.txt
```
## 1.2 .Data preparation
# extract 2-D keypoints from the sample videos
python utils/skeleton_extraction.py \
       --video_dir data/videos \
       --out_dir  data/keypoints

# generate dummy labels (5 classes, round-robin)
python utils/create_dummy_labels.py \
       --video_dir data/videos \
       --csv       data/labels/labels.csv
## 1.3 .Task 1 - Skeleton + LSTM
# train
python train/train_skeleton_model.py \
       --keypoints_dir data/keypoints \
       --labels_csv    data/labels/labels.csv

# inference (npy or video)
python inference/inference_task1.py \
       --input_path data/keypoints/sample.npy \
       --model_path best_model_task1.pth

## 1.4 Task 2 - SSL Encoder + LSTM
# self-supervised adapter pre-training
python train/pretrain_ssl_task2.py \
       --image_dir data/frames \
       --out_path  model_weights/dino_adapter.pth

# train LSTM head on videos
python train/train_video_model_task2_3.py \
       --model_type lstm \
       --video_dir  data/videos \
       --labels_csv data/labels/labels.csv

## 1.5 Task 3 - SSL Encoder + Transformer
# train Transformer head
python train/train_video_model_task2_3.py \
       --model_type transformer \
       --video_dir  data/videos \
       --labels_csv data/labels/labels.csv

## 1.6 .Unified inference (Task 2 & 3)
# choose --model_type lstm | transformer
python inference/inference_task2_3.py \
       --model_type transformer \
       --model_path best_video_model_tx.pth \
       --video_path data/videos/sample.avi

All scripts support --help for additional flags (sequence length, adapter layers, etc.).
