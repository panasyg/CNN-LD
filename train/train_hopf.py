# train/train_hopf.py

import sys
import os

# Додаємо кореневу папку проєкту в sys.path, щоб імпорти працювали коректно
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net import get_autoencoder
from hopf.hopf_model import HopfCNN
from datasets.mines_dataset import MinesDataset

# ==== Конфігурація ====
PATCH_SIZE = (1, 64, 64)       # (C, H, W)
FEATURE_DIMS = 8               # Число каналів latent map
HOPF_CHANNELS = 64             # Вихідні канали HopfLayer
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3

# ==== Пристрій ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ==== Підготовка моделей ====
# Створюємо Autoencoder і окремо енкодер
autoencoder, encoder = get_autoencoder(PATCH_SIZE, feature_dims=FEATURE_DIMS)
autoencoder.to(DEVICE)
encoder.to(DEVICE)

# Якщо вже є збережена модель автоенкодера, завантажуємо її:
ae_path = os.path.join("saved_models", "autoencoder.pth")
if os.path.exists(ae_path):
    autoencoder.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    print(f"[INFO] Loaded pretrained Autoencoder from {ae_path}")
autoencoder.eval()  # заморожуємо автоенкодер

# Створюємо HopfCNN
hopf_net = HopfCNN(in_channels=FEATURE_DIMS, hopf_channels=HOPF_CHANNELS).to(DEVICE)

# ==== Оптимізатор та функція втрат ====
criterion = nn.BCELoss()
optimizer = optim.Adam(hopf_net.parameters(), lr=LR)

# ==== Датасети та лоадери ====
train_dataset = MinesDataset(split='train')
val_dataset   = MinesDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==== Тренувальна петля ====
for epoch in range(1, EPOCHS + 1):
    hopf_net.train()
    total_loss = 0.0

    for imgs, masks in train_loader:
        imgs  = imgs.to(DEVICE)     # (B, 1, H, W)
        masks = masks.to(DEVICE)    # (B, 1, H, W)

        # Отримуємо фічі з енкодера
        with torch.no_grad():
            _, feats = autoencoder(imgs)

        # Передбачення HopfCNN
        preds = hopf_net(feats)

        # Втрата
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

    # ==== Валідація ====
    hopf_net.eval()
    total_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            _, feats = autoencoder(imgs)
            preds = hopf_net(feats)
            preds_bin = (preds > 0.5).float()

            # IoU
            intersection = (preds_bin * masks).sum(dim=(1,2,3))
            union = preds_bin.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - intersection
            iou = (intersection / (union + 1e-6)).mean().item()
            total_iou += iou * imgs.size(0)

    avg_iou = total_iou / len(val_loader.dataset)
    print(f"[Epoch {epoch}/{EPOCHS}] Val  IoU:  {avg_iou:.4f}")

# ==== Збереження моделі ====
os.makedirs("saved_models", exist_ok=True)
torch.save(hopf_net.state_dict(), os.path.join("saved_models", "hopf_net.pth"))
print(f"[INFO] Saved HopfCNN to saved_models/hopf_net.pth")
