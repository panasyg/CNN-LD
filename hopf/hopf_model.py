import torch
import torch.nn as nn
from hopf.hopf_layer import HopfLayer

class HopfCNN(nn.Module):
    def __init__(self, in_channels=32, hopf_channels=64):
        super(HopfCNN, self).__init__()

        # Hopf шар — витягує фазові ознаки
        self.hopf = HopfLayer(in_channels, hopf_channels)

        # Класифікаційна частина — умовна сегментація
        self.classifier = nn.Sequential(
            nn.Conv2d(hopf_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1),  # одноканальна карта ймовірності
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)
        phase_maps = self.hopf(x)         # (B, hopf_channels, H, W)
        prob_map = self.classifier(phase_maps)  # (B, 1, H, W)
        return prob_map

