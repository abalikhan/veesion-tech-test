# Veesion Technical Test · Abid Ali

*Concise end-to-end pipeline for temporal human-gesture **detection***

---

## 1 Quick start (≈ 3 min)

### 1.1 Environment

```bash
conda create -n veesion python=3.10 -y
conda activate veesion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
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

| Portion                                                                                | Origin               |
| -------------------------------------------------------------------------------------- | -------------------- |
| Arg-parsers, boiler-plate loops including train and inference, Dinov2 usage from HuggingFace                                                        | **ChatGPT scaffold** |
| Design choices, Model modelling including custom adapters, utils, dataloaders, ideas, final README | **Manual**           |

Inline comments are tagged `# LLM scaffold` or `# Manual`.

---

## 3 Design choices (fixed framework)

| Block           | Implementation                                          | Detection benefit                     |
| --------------- | ------------------------------------------------------- | ------------------------------------- |
| 2-D skeletons   | MediaPipe (body + hands)                                | Fast demo; fine hand motion preserved |
| SSL encoder     | DINOv2 frozen + 3 × 64-dim adapters (layers 0/5/11)     | Domain adapts with < 0.5 % new params |
| Temporal head A | 1-layer **LSTM** (hidden 256)                           | Low‑latency, online                   |
| Temporal head B | 2-layer **TransformerEncoder** (8 heads, sinusoidal PE) | Long-range context                    |
| Detector logic  | 16‑frame sliding window (stride 4)                      | Clip classifier → gesture boundaries  |

---

## 4 Next steps (with more data)

* **Pose quality** – ViTPose / PCIE‑Pose for robust finger joints
* **Encoder** – VideoMAE‑v2 or VideoMamba for video-native SSL
* **Temporal** – Mamba or long-sequence Transformers for minute-long clips
* **Context** – RT‑DETR / YOLOv9 boxes for person‑object cues
* **Weak supervision** – HATNet, WS‑STRONG (CVPR 24/25) to reduce labeling cost
* **Multimodal** – Qwen‑VL / MAViL for language‑conditioned gesture search

---

*Questions? Open an issue or email – I’m happy to elaborate on any detail.*
