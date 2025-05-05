import torch
import torch.nn as nn
import math

class HopfLayer(nn.Module):
    """
    Осциляторний шар, що перетворює просторові карти ознак
    через фазову активацію sin(ω * x).
    """
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        # Згортка для локальних залежностей
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Параметр ω — частота осциляцій
        if omega_init == 'random':
            # Навчальні частоти
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        else:
            # Фіксовані лінійно розподілені частоти
            buf = torch.linspace(0, 2 * math.pi, out_channels)
            self.register_buffer('omega', buf)

    def forward(self, x):
        """
        x: тензор розміру (B, in_channels, H, W)
        повертає: (B, out_channels, H, W)
        """
        y = self.conv(x)
        # фазова активізація
        # omega: (out_channels,) → (1, out_channels, 1, 1)
        return torch.sin(y * self.omega.view(1, -1, 1, 1))
