import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.optim as optim
from torch.utils.data import DataLoader
from net import Autoencoder
from hopf.hopf_model import HopfCNN
from datasets.mines_dataset import MinesDataset  # зроби власний датасет тут
import os
from tqdm import tqdm

# ==== Конфігурація ====
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Датасет ====
train_dataset = MinesDataset(split='train')
val_dataset = MinesDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==== Моделі ====
autoencoder = Autoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load('saved_models/autoencoder.pth'))
autoencoder.eval()  # заморожуємо

hopf_net = HopfCNN(in_channels=32, hopf_channels=64).to(DEVICE)  # 32 — output AE

# ==== Оптимізація ====
criterion = nn.BCELoss()
optimizer = optim.Adam(hopf_net.parameters(), lr=LR)

# ==== Тренування ====
for epoch in range(EPOCHS):
    hopf_net.train()
    epoch_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = batch  # labels — маска мін
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            encoded_feats = autoencoder.encoder(imgs)  # отримати просторові фічі

        preds = hopf_net(encoded_feats)
        loss = criterion(preds, labels.unsqueeze(1))  # BCE expects (B,1,H,W)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(train_loader):.4f}")

    # ==== Валідація ====
    hopf_net.eval()
    total_iou = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            encoded_feats = autoencoder.encoder(imgs)
            preds = hopf_net(encoded_feats)
            preds_bin = (preds > 0.5).float()
            intersection = (preds_bin * labels.unsqueeze(1)).sum()
            union = preds_bin.sum() + labels.unsqueeze(1).sum() - intersection
            iou = intersection / union.clamp(min=1e-6)
            total_iou += iou.item()

    print(f"[Val] Mean IoU: {total_iou / len(val_loader):.4f}")

# ==== Збереження ====
os.makedirs('saved_models', exist_ok=True)
torch.save(hopf_net.state_dict(), 'saved_models/hopf_net.pth')

