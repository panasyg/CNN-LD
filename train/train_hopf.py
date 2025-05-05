import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from net import get_autoencoder
from hopf.hopf_model import HopfCNN
from datasets.mines_dataset import MinesDataset

def main():
    # Конфігурація
    PATCH_SIZE    = (1, 10, 64, 64)   # (C, D, H, W)
    FEATURE_DIMS  = 16
    HOPF_CHANNELS = 32
    EPOCHS        = 20
    BATCH_SIZE    = 4
    LR            = 1e-3

    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps"   if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {DEVICE}")

    # Моделі
    ae3d, encoder3d = get_autoencoder(PATCH_SIZE, feature_dims=FEATURE_DIMS)
    ae3d.to(DEVICE).eval()
    hopf_net = HopfCNN(in_ch=FEATURE_DIMS, hopf_ch=HOPF_CHANNELS).to(DEVICE)

    # Оптимізатор і втрата
    optimizer = optim.Adam(hopf_net.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # Датасети
    train_ds = MinesDataset(split='train')
    val_ds   = MinesDataset(split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Тренування
    for epoch in range(1, EPOCHS+1):
        hopf_net.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad():
                _, feats = ae3d(imgs)
            preds = hopf_net(feats)
            loss  = criterion(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        print(f"Epoch {epoch}/{EPOCHS} Train Loss: {total_loss/len(train_loader.dataset):.4f}")

        # Валідація
        hopf_net.eval()
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                _, feats = ae3d(imgs)
                preds = hopf_net(feats)
                binp = (preds>0.5).float()
                inter = (binp * masks).sum((1,2,3))
                union = binp.sum((1,2,3)) + masks.sum((1,2,3)) - inter
                total_iou += (inter/(union+1e-6)).mean().item() * imgs.size(0)
        print(f"Epoch {epoch}/{EPOCHS} Val IoU: {total_iou/len(val_loader.dataset):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(hopf_net.state_dict(), "saved_models/hopf_net.pth")
    print("Saved HopfCNN3D.")

if __name__=="__main__":
    torch.multiprocessing.freeze_support()
    main()
