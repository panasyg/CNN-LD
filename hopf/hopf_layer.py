import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega_init='random'):
        super(HopfLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Звичайна згортка
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Частоти осциляцій ω — або випадкові, або фіксовані
        if omega_init == 'random':
            self.omega = nn.Parameter(torch.rand(out_channels) * 2 * math.pi)
        elif omega_init == 'fixed':
            self.register_buffer('omega', torch.linspace(0, 2 * math.pi, out_channels))

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        conv_out = self.conv(x)  # (B, C_out, H, W)
        
        # Імітація осциляторної активації через синус функцію
        # Перетворення просторових карт в осциляторний простір
        out = torch.sin(conv_out * self.omega.view(1, -1, 1, 1))

        return out

