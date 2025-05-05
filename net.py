# net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"   if torch.backends.mps.is_available() else
                      "cpu")

print(f"Using device: {DEVICE}")

class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1, feature_dims=8):
        super().__init__()
        # 3D Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3,6,6), padding=(1,2,2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, feature_dims, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # 3D Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_dims, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 16, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), output_padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, in_channels, kernel_size=(3,6,6), padding=(1,2,2)),
            nn.Tanh(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        rec    = self.decoder(latent)
        return rec, latent

def get_autoencoder(patch_size, feature_dims=8):
    """
    patch_size: (C, D, H, W)
    """
    in_ch = patch_size[0]
    model = Autoencoder3D(in_channels=in_ch, feature_dims=feature_dims).to(DEVICE)
    return model, model.encoder

if __name__=="__main__":
    dummy = torch.randn(1,1,10,64,64).to(DEVICE)
    ae3d, enc3d = get_autoencoder3d((1,10,64,64))
    rec, lat = ae3d(dummy)
    print("latent:", lat.shape, "recon:", rec.shape)
