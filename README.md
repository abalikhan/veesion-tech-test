# Veesion Technical Test – Abid Ali  
_Building a concise, end-to-end pipeline for temporal human gesture classification_

---

## 1 .Quick setup

```bash
# Create environment (example)
conda create -n veesion-env python=3.10 -y
conda activate veesion-env

# Install PyTorch with CUDA 11.8 (works on RTX 3000)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project requirements
pip install -r requirements.txt
```
## 2 .How to run
```bash
| Step | Command (defaults) | Result |
|------|--------------------|--------|
| Skeleton extraction | `python utils/skeleton_extraction.py --video_dir data/videos --out_dir data/keypoints` | MediaPipe keypoints → `.npy` |
| Dummy labels | `python utils/create_dummy_labels.py --video_dir data/videos --csv data/labels/labels.csv` | `labels.csv` |
| **Task 1 – train LSTM** | `python train/train_skeleton_model.py --keypoints_dir data/keypoints --labels_csv data/labels/labels.csv` | `best_model_task1.pth` |
| SSL pre-train adapters | `python train/pretrain_ssl_task2.py --image_dir data/frames` | `model_weights/dino_adapter.pth` |
| **Task 2 – video LSTM** | `python train/train_video_model_task2_3.py --model_type lstm --video_dir data/videos --labels_csv data/labels/labels.csv` | `best_video_model_lstm.pth` |
| **Task 3 – video Tx** | `python train/train_video_model_task2_3.py --model_type transformer --video_dir data/videos --labels_csv data/labels/labels.csv` | `best_video_model_tx.pth` |
| Inference | Task 1: `python inference/inference_task1.py …`<br>Task 2/3: `python inference/inference_task2_3.py --model_type transformer …` | Class + probabilities |

All scripts expose `--help` for further flags (sequence length, adapter layers, etc.).

```

## 2 · LLM vs. manual code

| Component | Origin |
|-----------|--------|
| Extraction boiler-plate, arg-parser skeletons, initial SimCLR loop | **ChatGPT scaffold** |
| Adapter class, masking logic, weight-init utils, sliding-window detector, README text | **Manual** |
| Hyper-parameters, data split, smoothing suggestions, compute table | **Manual** |

Inline comments mark **`# LLM scaffold`** or **`# Manual`**.

---

## 3 · Design choices (within mandatory framework)

> Required by the brief: **(a)** 2-D skeletons **(b)** SSL-pretrained 2-D frame encoder **(c)** LSTM temporal model **(d)** Transformer temporal model.  
> The points below explain *how* each mandatory piece was realised and combined.

1. **2-D skeletons via MediaPipe** – Fast CPU inference keeps the repo light. All 33 body + 42 hand joints are kept so fine hand motion is available to downstream models.  
2. **SSL frame encoder** – DINOv2 is frozen; three 64-dim bottleneck adapters are trained with SimCLR. This satisfies the “pretrain SSL” requirement while limiting trainable params ≈ 0.5 %.  
3. **Temporal head A: LSTM** – Single layer, hidden = 256, dropout 0.3; padding mask ignores zero-filled frames. Matches the “LSTM or GRU” requirement and gives online latency.  
4. **Temporal head B: Transformer** – Two-layer `nn.TransformerEncoder`, 8 heads, sinusoidal positions. Flag toggles CLS-token vs mean pooling. Meets the “replace LSTM with Transformer” clause.  
5. **Detection logic** – A 16-frame sliding window (stride 4) turns the classifier into an online gesture **detector** without altering the mandated architecture.

---

## 4 · If I had more time / data

| Area | Next step & rationale |
|------|-----------------------|
| Pose quality | Swap MediaPipe for **ViTPose** or **PCIE-Pose** – better finger joints, fewer dropouts |
| Frame encoder | Pretrain adapters on larger CCTV corpus; test **VideoMAE-v2** or **VideoMamba** for video-native features |
| Temporal model | Explore memory-linear models (Mamba) or longer Transformers for minute-long clips |
| Context fusion | Add person/object boxes from **RT-DETR / YOLOv9** for interaction-aware gesture parsing |
| Weak supervision | Adopt architectures like **HATNet** or **WS-STRONG** (CVPR 24/25) to reduce labeling cost |
| Multimodal | Stack **Qwen-VL** or **MAViL** for language-conditioned gesture search |

---

_Questions? I’m happy to elaborate on any implementation detail or design choice._
