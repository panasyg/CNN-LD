import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class HopfLayer(nn.Module):
    def init(self, 
                 in_channels: int,
                 out_channels: int,
                 dt: float = 0.025,
                 T: int = 8,
                 alpha: float = 1.0,
                 init_freq: float = 1.0,
                 trainable_params: bool = True):
        """
        Осциляторний шар Хопфа для попередньої обробки зображень
        
        Параметри:
            in_channels: кількість вхідних каналів
            out_channels: кількість осциляторів (вихідних каналів)
            dt: крок інтегрування
            T: кількість ітерацій динаміки
            alpha: коефіцієнт зв'язку
            init_freq: початкова частота осциляторів
            trainable_params: чи навчати параметри динаміки
        """
        super().init()
        
        # Параметри динаміки
        self.dt = dt
        self.T = T
        self.alpha = alpha
        
        # Навчаємі параметри
        if trainable_params:
            self.omega = nn.Parameter(torch.empty(out_channels))
            self.gamma = nn.Parameter(torch.empty(out_channels))
            self.W = nn.Parameter(torch.empty(out_channels, in_channels))
        else:
            self.register_buffer('omega', torch.empty(out_channels))
            self.register_buffer('gamma', torch.empty(out_channels))
            self.register_buffer('W', torch.empty(out_channels, in_channels))
            
        # Ініціалізація параметрів
        self._init_parameters(init_freq)
        
        # Нормалізація
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        
        # Кеш для проміжних станів (для аналізу)
        self.state_cache = None

    def _init_parameters(self, init_freq):
        """Ініціалізація параметрів осциляторів"""
        nn.init.uniform_(self.omega, 
                         init_freq * 0.9 * 2*np.pi, 
                         init_freq * 1.1 * 2*np.pi)
        nn.init.constant_(self.gamma, -0.05)
        nn.init.xavier_normal_(self.W, gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вхід: 
            x - тензор зображень [batch, in_channels, height, width]
        Вихід:
            Тензор амплітуд [batch, out_channels, height, width]
        """
        batch, _, h, w = x.shape
        device = x.device
        
        # Ініціалізація комплексних станів
        z = torch.zeros(batch, self.out_channels, h, w, 
                        dtype=torch.cfloat, device=device)
        
        # Осциляторна динаміка
        for t in range(self.T):
            # Обчислення динаміки
            r = torch.abs(z)
            dz = (self.gamma[None,:,None,None] + 
                  1j*self.omega[None,:,None,None] - 
                  self.alpha * r**2) * z
            
            # Вхідний сигнал
            input_term = torch.einsum('oi,bihw->bohw', self.W, x)
            
            # Оновлення стану
            z = z + self.dt * (dz + input_term)
            
            # Зберігання станів для аналізу
            if self.training and t == self.T-1:
                self.state_cache = {
                    'amplitude': r.detach(),
                    'phase': torch.angle(z).detach()
                }
        
        # Повертаємо амплітуду з нормалізацією
        return self.norm(torch.abs(z))

    @property
    def out_channels(self) -> int:
        return self.omega.shape[0]
    
    def get_oscillator_params(self) -> dict:
        """Повертає поточні параметри осциляторів"""
        return {
            'frequencies': self.omega.detach().cpu() / (2*np.pi),
            'damping': self.gamma.detach().cpu(),
            'weights': self.W.detach().cpu()
        }
def visualize_dynamics(self):
        """Візуалізація динаміки (вимагає matplotlib)"""
        if self.state_cache is None:
            raise ValueError("No cached states. Run forward pass first.")
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Амплітуда
        amp = self.state_cache['amplitude'][0].mean(dim=(1,2)).cpu()
        ax1.plot(amp.numpy())
        ax1.set_title('Середня амплітуда по каналам')
        ax1.set_xlabel('Канал')
        ax1.set_ylabel('Амплітуда')
        
        # Фаза
        phase = self.state_cache['phase'][0].mean(dim=(1,2)).cpu()
        ax2.plot(phase.numpy())
        ax2.set_title('Середня фаза по каналам')
        ax2.set_xlabel('Канал')
        ax2.set_ylabel('Фаза (рад)')
        
        plt.tight_layout()
        return fig
