# net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Визначаємо пристрій: GPU-NVIDIA (якщо є), Metal (MPS на Apple Silicon) або CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():  # Metal Performance Shaders на Mac M1/M2/M3
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

class Autoencoder(nn.Module):
    """
    PyTorch-реалізація CNN-автоенкодера за зразком Keras-функцій Auto1/Auto2/Auto3.
    Повертає як повну модель, так і окремий енкодер.
    """
    def __init__(self, in_channels=1, feature_dims=8):
        """
        :param in_channels: число каналів вхідного зображення (наприклад, 1 для GPR)
        :param feature_dims: число каналів у латентному представленні (encoder output)
        """
        super().__init__()
        # Енкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=1, padding=2),  # аналог Conv2D(16,6x6)
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),  # 16×(5×5), stride=2
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 16×(4×4), stride=2
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 16×(3×3), stride=2
            nn.ReLU(inplace=True),

            nn.Conv2d(16, feature_dims, kernel_size=1, stride=2, padding=0),  # latent map
            nn.ReLU(inplace=True),
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dims, 16, kernel_size=2, stride=2, padding=0),  # 2×2
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3×3
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),  # 4×4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # 5×5
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, in_channels, kernel_size=6, stride=1, padding=2),  # 6×6
            nn.Tanh(),  # як в Keras-версії
        )

    def forward(self, x):
        """
        :param x: вхідний тензор shape=(B, in_channels, H, W)
        :return: (reconstructed, latent_features)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def get_autoencoder(patch_size, feature_dims=8):
    """
    Фабрика: створює Autoencoder під конкретний розмір патчу.
    :param patch_size: tuple (C, H, W)
    :param feature_dims: число каналів latent map
    :return: (autoencoder_model, encoder_module)
    """
    in_channels = patch_size[0]
    model = Autoencoder(in_channels=in_channels, feature_dims=feature_dims).to(DEVICE)
    return model, model.encoder


if __name__ == "__main__":
    # Простісний тест архітектури
    dummy = torch.randn(1, 1, 64, 64).to(DEVICE)
    ae, enc = get_autoencoder((1, 64, 64), feature_dims=8)
    recon, lat = ae(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Latent shape: {lat.shape}")
    print(f"Reconstructed shape: {recon.shape}")
