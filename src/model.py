import torch.nn as nn

class  DenoisingAutoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # (B, 64, 16, 16)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (B, 32, 16, 16)
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),  # (B, 32, 32, 32)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # (B, 3, 32, 32)
            nn.Tanh(),  # sortie dans [-1, 1]

        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z



