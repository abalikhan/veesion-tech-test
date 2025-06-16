# Veesion Technical Test · Abid Ali

*Concise end-to-end pipeline for temporal human-gesture **Recognition***

---

## 1 Quick start

### 1.1 Environment

```bash

conda create -n veesion python=3.10 -y
conda activate veesion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
# install repo as editable package (ensures clean imports)
pip install -e .

```

### 1.2 Data prep

```bash

python utils/skeleton_extraction.py \
       --video_dir data/videos \
       --out_dir   data/keypoints

python utils/create_dummy_labels.py \
       --video_dir data/videos \
       --csv       data/labels/labels.csv

```

### 1.3 Train / infer

| Task                    | Train command                                                                                        | Inference                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **1 Skeleton LSTM**     | `python train/train_skeleton_model.py`                                                               | `python inference/inference_task1.py`                            |
| **2 SSL + LSTM**        | `python train/pretrain_ssl_task2.py` → `python train/train_video_model_task2_3.py --model_type lstm` | `python inference/inference_task2_3.py --model_type lstm`        |
| **3 SSL + Transformer** | *(reuse adapters)* → `python train/train_video_model_task2_3.py --model_type transformer`            | `python inference/inference_task2_3.py --model_type transformer` |

> Run any script with `--help` for extra flags (sequence length, adapter layers, etc.).

---

## 2 LLM vs. manual code

| Portion | Origin |
|---------|--------|
| Arg-parsers and boiler-plate loops (train & inference).<br>Skeleton extraction and the initial README. | **ChatGPT scaffold** |
| Design choices & ideas and custom adapter coding.<br>Most utils/data-loaders and the final README. | **Manual** |   

Inline comments are tagged `# by LLM ` or `# Manual`.

---

## 3 Design choices
### Task 1 Skeleton Model + LSTM
1) I selected **MediaPipe** for skeleton extraction because of its speed, reliability, and minimal code complexity. This avoids installing and debugging heavyweight extractors (e.g. OpenPose, ViTPose). But there are more robust methods that can be utilized for this part such as ViTPose, PCIE-Pose or OpenGait.
2) **How it fits gesture detection?** Sliding a short-window LSTM over successive keypoint frames turns the clip-level model into a frame-time gesture detector, pinpointing when each action starts and ends.
### Task 2 + 3 Pretrain 2D Encoder in SSL encoder & temporal head
1) Adapting a powerful SSL backbone (DINOv2) with lightweight adapters (< 0.5 % params) gives rapid domain transfer to our gesture clips; the same strategy would work with MAE or newer video models like VideoMAE-v2 or VideoMamba.
2) I modeled basic adapters, but we can opt latest techniques such as IA^3, LoRA, AdapterFusion, or even temporal adapters from our Paper "AM Flow: Adapters for Temporal Processing in Action Recognition".
### How it fits gesture detection (Task 2 & 3)
1) Frame-level logits – Make the LSTM or Transformer output a score per frame (or short clip) rather than a single video label.
2) Sliding / chunked inference – Run the network on overlapping windows; threshold and group contiguous positives, then apply temporal-NMS to yield start–end segments.
3) Long-video scalability – Process long streams in chunks with hidden-state carry-over (LSTM) or memory-efficient attention blocks (e.g., sparse/segmental Transformer); this keeps detection feasible on hour-long footage without losing context.

| Block           | Implementation                                          | Benefit                     |
| --------------- | ------------------------------------------------------- | ------------------------------------- |
| 2-D skeletons   | MediaPipe (body + hands)                                | Fast demo; fine hand motion preserved |
| SSL encoder     | DINOv2 frozen + 3 × 64-dim adapters (layers 0/5/11)     | Domain adapts with < 0.5 % new params |
| Temporal head A | 1-layer **LSTM** (hidden 256)                           | Low‑latency, online                   |
| Temporal head B | 2-layer **TransformerEncoder** (8 heads, sinusoidal PE) | Long-range context                    |

---

## 4 Next steps (with more data & time) 

* **Pose quality** – ViTPose / PCIE‑Pose for robust finger joints. Focuses on most important keypoints (especially for surveillance task) such as hands, fingers. 
* **Encoder** – VideoMAE‑v2 or VideoMamba for video-native SSL
* **Temporal** – Mamba or long-sequence Transformers for minute-long clips, learnable or relative temporal positional embeddings.
* **Context** – RT‑DETR / YOLOv9 boxes for person‑object cues. Focuses the model on active people/hands/objects, filtering irrelevant motion. Captures interactions (e.g., picking up objects, waving, pointing).
* **Weak supervision** – HATNet, WS‑STRONG (CVPR 24/25) to reduce labeling cost
* **Multimodal** – Qwen‑VL for language‑conditioned gesture search

---

