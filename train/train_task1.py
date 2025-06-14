from torch.utils.data import DataLoader, random_split
import torch
from datasets.skeleton_dataset import SkeletonDataset
from models.skeleton_lstm_model import SkeletonLSTMClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# This file assumes the repo is installed in editable mode using:
#   pip install -e .
# This avoids fragile sys.path hacks and ensures clean modular imports.


# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 5
VAL_SPLIT = 0.2
SEED = 42

# Load dataset and split to Trian & Val
dataset = SkeletonDataset("./data/keypoints/", label_csv="data/labels/labels.csv")
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu exists
model = SkeletonLSTMClassifier(num_classes=NUM_CLASSES).to(device) # load the model
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# Validation Function
def validate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

# Training Loop
best_val_acc = 0.0  # keep track of the best model and save it later for training
save_path = "best_model_task1.pth"  # To save the model for later use in inference

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        out = model(data)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    avg_train_loss = total_loss / len(train_loader)

    val_loss, val_acc = validate(model, val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save only if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Update the best model (val acc: {val_acc:.4f}), saved at {save_path}")