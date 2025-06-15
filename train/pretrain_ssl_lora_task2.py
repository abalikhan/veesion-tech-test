import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.ssl_dataset import SSLDataset
from models.dino_variable_adapters import DinoV2VariableAdapters
from utils.ssl_utils import ssl_transform_1, ssl_transform_2, ssl_collate_fn, parse_layers, contrastive_loss, set_seed
import os
import argparse

def train_ssl(
    image_dir,
    batch_size,
    epochs,
    lr,
    out_path,
    adapter_dim,
    adapter_layers,
    seed
):
    set_seed(seed)  # Set the random seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    dataset = SSLDataset(image_dir, ssl_transform_1, ssl_transform_2)
    loader = DataLoader(dataset,
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=ssl_collate_fn,
                        drop_last=True)

    # Model & optimizer
    model = DinoV2VariableAdapters(adapter_dim=adapter_dim, adapter_layers=adapter_layers).to(device)
    # Only optimize adapter parameters
    adapter_params = list(model.adapters.parameters())
    optimizer = torch.optim.Adam(adapter_params, lr=lr)

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for img1, img2 in loader:
            # Prepare inputs
            data1 = model.processor(img1, return_tensors="pt")
            data2 = model.processor(img2, return_tensors="pt")
            pixel_values1 = data1["pixel_values"].to(device)
            pixel_values2 = data2["pixel_values"].to(device)

            # Forward
            feat1 = model(pixel_values1)
            feat2 = model(pixel_values2)

            # Contrastive loss
            loss = contrastive_loss(feat1, feat2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

    # Save the adapted model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"SSL pretraining done, weights saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Pretraining with DINOv2 Custom Adapters")
    parser.add_argument('--image_dir', type=str, default="./data/frames")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_path', type=str, default="model_weights/dino_adapter.pth")
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--adapter_layers', type=parse_layers, default="0, 5, 11",
                        help="Comma-separated indices of transformer blocks for adapter injection, e.g. '0,5,11'")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Parse adapter_layers if it's a string
    adapter_layers = args.adapter_layers
    if isinstance(adapter_layers, str):
        adapter_layers = parse_layers(adapter_layers)
    elif isinstance(adapter_layers, int):
        adapter_layers = [adapter_layers]

    train_ssl(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        out_path=args.out_path,
        adapter_dim=args.adapter_dim,
        adapter_layers=adapter_layers,
        seed=args.seed
    )