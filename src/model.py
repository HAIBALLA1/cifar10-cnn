import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=3, c1=32, c2=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z


def build_model(cfg):
    """
    Factory Hydra: construit le modèle à partir de cfg.model
    """
    mcfg = cfg.model
    return DenoisingAutoencoder(
        in_channels=int(mcfg.in_channels),
        c1=int(mcfg.c1),
        c2=int(mcfg.c2),
    )
