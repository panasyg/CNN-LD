import torch
import torch.nn as nn
import torch.nn.functional as F
from .hopf_layer import HopfLayer
from typing import Tuple, Optional

class HopfCNN(nn.Module):
    def init(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_channels: int = 32,
                 hopf_params: Optional[dict] = None):
        """
        Інтегрована модель CNN з осциляторним шаром Хопфа
        
        Параметри:
            in_channels: кількість вхідних каналів (3 для RGB)
            num_classes: кількість класів для класифікації
            base_channels: базовий розмір каналів у шарах
            hopf_params: параметри для HopfLayer (див. hopf_layer.py)
        """
        super().init()
        
        # Конфігурація осциляторного шару
        default_hopf_params = {
            'in_channels': in_channels,
            'out_channels': base_channels,
            'dt': 0.03,
            'T': 6,
            'alpha': 0.8,
            'trainable_params': True
        }
        if hopf_params is not None:
            default_hopf_params.update(hopf_params)
        
        # Осциляторний шар
        self.hopf_layer = HopfLayer(**default_hopf_params)
        
        # CNN-екстрактор ознак
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)))
        
        
        # Класифікатор
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*8, base_channels*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels*4, num_classes)
        )
        
        # Ініціалізація ваг
        self._initialize_weights()

    def _initialize_weights(self):
        """Ініціалізація ваг згорткових шарів"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Вхід:
            x - тензор зображень [batch, in_channels, height, width]
        Вихід:
            logits - тензор класифікації [batch, num_classes]
            dynamics - словник з динамічними параметрами (якщо train mode)
        """
        dynamics_info = None
        
        # Осциляторна обробка
        x = self.hopf_layer(x)
        
        # Збереження динаміки для аналізу
        if self.training and hasattr(self.hopf_layer, 'state_cache'):
            dynamics_info = {
                'amplitude': self.hopf_layer.state_cache['amplitude'].mean(dim=(0,2,3)),
                'phase': self.hopf_layer.state_cache['phase'].mean(dim=(0,2,3)),
                'params': self.hopf_layer.get_oscillator_params()
            }
        
        # Виділення ознак
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # Класифікація
        logits = self.classifier(features)
        
        return logits, dynamics_info

    def get_feature_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Отримання feature maps з проміжного шару
        
        Параметри:
            x: вхідний тензор
            layer_idx: індекс шару (0 - hopf, 1-3 - conv blocks)
        Вихід:
            Тензор feature maps [batch, channels, h, w]
        """
        layers = [
            self.hopf_layer,
            *[layer for layer in self.feature_extractor.children()]
        ]
        
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Invalid layer index. Must be between 0 and {len(layers)-1}")
        
        with torch.no_grad():
            for i, layer in enumerate(layers[:layer_idx+1]):
                x = layer(x)
                if i == 0:  # Після hopf шару
                    x = torch.abs(x)
        return x