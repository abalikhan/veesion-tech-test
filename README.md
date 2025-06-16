content = """# Veesion Technical Test · Abid Ali  
_Concise end‑to‑end pipeline for temporal human‑gesture **detection**_

---

## 1 · Quick start (≈ 3 min)

### 1.1 Environment

```bash
conda create -n veesion python=3.10 -y && conda activate veesion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
### 1.2 Data prep
python utils/skeleton_extraction.py \
       --video_dir data/videos \
       --out_dir   data/keypoints

python utils/create_dummy_labels.py \
       --video_dir data/videos \
       --csv       data/labels/labels.csv
### 1.3 Train & test

Task	Train	Inference
1 Skeleton LSTM	python train/train_skeleton_model.py	python inference/inference_task1.py
2 SSL + LSTM	python train/pretrain_ssl_task2.py → python train/train_video_model_task2_3.py --model_type lstm	python inference/inference_task2_3.py --model_type lstm
3 SSL + Transformer	(reuse adapters) → python train/train_video_model_task2_3.py --model_type transformer	python inference/inference_task2_3.py --model_type transformer

Every script accepts --help for extra flags (sequence length, adapter layers, etc.).

## 2 · LLM vs. manual code

Portion	Origin
Arg‑parsers, boiler‑plate loops	ChatGPT scaffold
Adapter class, masking logic, weight‑init utils, sliding‑window detector, final README	Manual

Inline comments mark # LLM scaffold or # Manual.

## 3 · Design choices (fixed framework)

Mandatory block	Implementation	Detection benefit
2‑D skeletons	MediaPipe (body + hands)	Fast demo; preserves fine hand motion
SSL frame encoder	DINOv2 frozen + 3 × 64‑dim adapters (layers 0/5/11)	Domain adapts with < 0.5 % new params
Temporal head A	1‑layer LSTM (hidden 256)	Low‑latency, online
Temporal head B	2‑layer TransformerEncoder (8 heads, sinusoidal PE)	Long‑range context
Detector logic	16‑frame sliding window (stride 4)	Clip classifier → gesture boundaries

## 4 · What I’d improve with more time/data

Pose quality – ViTPose / PCIE‑Pose ⇒ better finger joints

Encoder – VideoMAE‑v2 / VideoMamba ⇒ video‑native SSL

Temporal – Mamba / long‑seq Transformer for minute‑long clips

Context – RT‑DETR / YOLOv9 boxes for interaction cues

Weak supervision – HATNet, WS‑STRONG (CVPR 24/25) to cut labeling cost

Multimodal – Qwen‑VL / MAViL for language‑conditioned search


