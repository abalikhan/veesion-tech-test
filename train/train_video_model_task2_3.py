import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

from datasets.video_dataset import VideoDataset
from models.video_lstm_customDinov2 import VideoLSTMCustomDinov2
from models.video_transformer_customDinov2 import VideoTransformerDinov2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss/len(loader), accuracy_score(all_labels, all_preds)

def main(args):
    # Reproducibility and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + split
    dataset = VideoDataset(
        video_dir=args.video_dir,
        label_csv=args.labels_csv,
        seq_len=args.seq_len
    )
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Model selection
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
            num_layers=args.trans_layers,
            num_heads=args.trans_heads,
            dropout=args.dropout,
            use_cls_token=args.use_cls_token,
            device=device,
            adapter_dim=args.adapter_dim,
            adapter_layers=args.adapter_layers,
        )
    model.to(device)

    # Optimizer & loss (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(all_labels, all_preds)
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val   Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_path = args.save_path + '_' + args.model_type + '.pth'
            torch.save(model.state_dict(), save_model_path)
            print(f" New best model (val_acc={val_acc:.4f}), saved to {save_model_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Train Video Model (LSTM or Transformer)")
    # Data & splits
    p.add_argument('--video_dir',    type=str, default="./data/videos")
    p.add_argument('--labels_csv',   type=str, default="./data/labels/labels.csv")
    p.add_argument('--seq_len',      type=int, default=8,
                   help="Fix #frames per video (truncate/pad); default = full")
    p.add_argument('--val_split',    type=float, default=0.2)
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--seed',         type=int, default=42)

    # Model choice
    p.add_argument('--model_type',   choices=['lstm','transformer'], default='transformer')
    # Common
    p.add_argument('--num_classes',  type=int, default=5)
    p.add_argument('--dropout',      type=float, default=0.3)
    p.add_argument('--backbone_ckpt',type=str, default="model_weights/dino_adapter.pth",
                   help="Path to our custom Dinov2 weights")
    p.add_argument('--adapter_dim',  type=int, default=64)
    p.add_argument('--adapter_layers', type=lambda s: [int(x) for x in s.split(',')],
                   default="0,5,11", help="Comma-list of adapter layer indices")
    # LSTM-specific
    p.add_argument('--lstm_hidden',  type=int, default=256)
    p.add_argument('--lstm_layers',  type=int, default=1)

    # Transformer-specific
    p.add_argument('--trans_layers', type=int, default=1)
    p.add_argument('--trans_heads',  type=int, default=8)
    p.add_argument('--use_cls_token',type=bool, default=True)

    # Training
    p.add_argument('--epochs',       type=int, default=10)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--save_path',    type=str, default="./model_weights/best_video_model")

    args = p.parse_args()
    main(args)
