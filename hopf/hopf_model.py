import torch
import torch.nn as nn
from hopf.hopf_layer import HopfLayer

class HopfCNN(nn.Module):
    """
    Проста сегментуюча CNN, що спочатку витягує фазові ознаки через HopfLayer,
    а потім перетворює їх у карту ймовірностей класу (міна/немає міни).
    """
    def __init__(self, in_channels=8, hopf_channels=32):
        super(HopfCNN, self).__init__()
        # Осциляторний шар
        self.hopf = HopfLayer(in_channels, hopf_channels)
        # Класифікаційний блок
        self.classifier = nn.Sequential(
            nn.Conv2d(hopf_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: фічі з автоенкодера, тензор розміру (B, in_channels, H, W)
        повертає: карта ймовірностей (B, 1, H, W)
        """
        phase = self.hopf(x)
        prob  = self.classifier(phase)
        return prob
