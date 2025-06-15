import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from models.video_lstm_customDinov2 import VideoLSTMCustomDinov2
from models.video_transformer_customDinov2 import VideoTransformerDinov2
from utils.preprocess_video import preprocess_video

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize processor (image transforms)
    processor = AutoImageProcessor.from_pretrained(
        args.backbone_pretrained if args.model_type == "transformer" else args.backbone_pretrained,
        trust_remote_code=True
    )

    # Build model
    if args.model_type == "lstm":
        model = VideoLSTMCustomDinov2(
            dinov2_ckpt_path=args.backbone_ckpt,
            num_classes=args.num_classes,
            adapter_dim=args.adapter_dim,
            adapter_layers=args.adapter_layers,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            device=device
        )
    else:
        model = VideoTransformerDinov2(
            dinov2_ckpt_path=args.backbone_ckpt,
            num_classes=args.num_classes,
            adapter_dim=args.adapter_dim,
            adapter_layers=args.adapter_layers,
            num_heads=args.trans_heads,
            num_layers=args.trans_layers,
            use_cls_token=args.use_cls_token,
            dropout=args.dropout,
            device=device
        )

    model.to(device) # keep model on gpu
    # Load weights if they exists
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
    
    # keep it in test mode
    model.eval()

    # Preprocess video
    pixel_seq = preprocess_video(args.video_path, processor, args.seq_len)
    pixel_seq = pixel_seq.unsqueeze(0).to(device)  # [1, T, C, H, W]

    # Inference
    with torch.no_grad():
        logits = model(pixel_seq)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    print(f"Video: {args.video_path}")
    print(f"Predicted class: {pred}")
    print(f"Probabilities: {probs.round(3)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Model Inference")
    # Model selection
    parser.add_argument('--model_type', choices=['lstm','transformer'], default= 'transformer')
    parser.add_argument('--model_path', type=str, help="Path to saved model.pth")
    parser.add_argument('--video_path', type=str, default='./data/videos' ,help="Input video file (.avi/.mp4)")
    parser.add_argument('--seq_len', type=int, default=8,
                        help="Fix sequence length (truncate/pad); default=full length")
    # Common
    parser.add_argument('--backbone_ckpt', type=str, default="./model_weights/dino_adapter.pth",
                        help="Checkpoint for DINOv2+adapter backbone (for both models)")
    parser.add_argument('--backbone_pretrained', type=str, default="facebook/dinov2-base",
                        help="HuggingFace backbone ID (for transformer model preprocessing)")
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--adapter_layers', type=lambda s: [int(x) for x in s.split(',')],
                        default="0,5,11", help="Comma-list of adapter layer indices")
    # LSTM-specific
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=1)
    # Transformer-specific
    parser.add_argument('--trans_heads', type=int, default=8)
    parser.add_argument('--trans_layers', type=int, default=2)
    parser.add_argument('--use_cls_token', type=bool, default=True)
    # Task-specific
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.3)

    args = parser.parse_args()
    # Parse adapter_layers if passed as string
    # if isinstance(args.adapter_layers, str):
    #     args.adapter_layers = [int(x) for x in args.adapter_layers.split(',')]
    main(args)