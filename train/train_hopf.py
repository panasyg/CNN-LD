# train/train_hopf.py

import sys
import os
# 1) Add project root to Python path immediately:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 2) Now these imports will work, because `net.py` is on sys.path
from net import get_autoencoder
from hopf.hopf_model import HopfCNN
from datasets.mines_dataset import MinesDataset

def main():
    # ——— Configuration ———
    PATCH_SIZE    = (1, 64, 64)
    FEATURE_DIMS  = 8
    HOPF_CHANNELS = 64
    EPOCHS        = 20
    BATCH_SIZE    = 16
    LR            = 1e-3

    # ——— Device setup ———
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[INFO] Using device: {DEVICE}")

    # ——— Models ———
    autoencoder, encoder = get_autoencoder(PATCH_SIZE, feature_dims=FEATURE_DIMS)
    autoencoder.to(DEVICE).eval()
    hopf_net = HopfCNN(in_channels=FEATURE_DIMS, hopf_channels=HOPF_CHANNELS).to(DEVICE)

    # ——— Optimizer & Loss ———
    optimizer = optim.Adam(hopf_net.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # ——— DataLoaders ———
    train_dataset = MinesDataset(split='train')
    val_dataset   = MinesDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    # ——— Training loop ———
    for epoch in range(1, EPOCHS + 1):
        hopf_net.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with torch.no_grad():
                _, feats = autoencoder(imgs)

            preds = hopf_net(feats)
            loss  = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

        # ——— Validation ———
        hopf_net.eval()
        total_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                _, feats = autoencoder(imgs)
                preds    = hopf_net(feats)
                preds_bin = (preds > 0.5).float()

                inter = (preds_bin * masks).sum(dim=(1,2,3))
                union = (preds_bin.sum(dim=(1,2,3)) +
                         masks.sum(dim=(1,2,3)) - inter)
                iou   = (inter / (union + 1e-6)).mean().item()
                total_iou += iou * imgs.size(0)

        avg_iou = total_iou / len(val_loader.dataset)
        print(f"[Epoch {epoch}/{EPOCHS}] Val  IoU:  {avg_iou:.4f}")

    # ——— Save model ———
    os.makedirs("saved_models", exist_ok=True)
    torch.save(hopf_net.state_dict(), "saved_models/hopf_net.pth")
    print("[INFO] Saved HopfCNN model.")

if __name__ == "__main__":
    # On macOS spawn start method, ensure main is protected
    torch.multiprocessing.freeze_support()
    main()
