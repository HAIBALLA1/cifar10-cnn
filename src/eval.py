# src/eval.py
import torch
import matplotlib.pyplot as plt

from src.data import get_cifar10_loaders
from src.model import DenoisingAutoencoder

from src.utils import add_noise



def show_images(x_noisy, x_recon, x_clean, n=5):
    """
    Affiche n images :
    - ligne 1 : bruitées
    - ligne 2 : reconstruites
    - ligne 3 : propres
    """
    x_noisy = x_noisy[:n].cpu()
    x_recon = x_recon[:n].cpu()
    x_clean = x_clean[:n].cpu()

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):
        # Noisy
        plt.subplot(3, n, i + 1)
        plt.imshow((x_noisy[i].permute(1, 2, 0) + 1) / 2)
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy")

        # Reconstructed
        plt.subplot(3, n, i + 1 + n)
        plt.imshow((x_recon[i].permute(1, 2, 0) + 1) / 2)
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed")

        # Clean
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow((x_clean[i].permute(1, 2, 0) + 1) / 2)
        plt.axis("off")
        if i == 0:
            plt.ylabel("Clean")

    plt.tight_layout()
    plt.show()


def evaluate(model_path="dae.pth", noise_std=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger les données de test
    _, test_loader = get_cifar10_loaders(
        batch_size=16,
        use_augmentation=False
    )

    # Charger le modèle
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Un batch de test
    x_clean, _ = next(iter(test_loader))
    x_clean = x_clean.to(device)

    x_noisy = add_noise(x_clean, noise_std)

    with torch.no_grad():
        x_recon = model(x_noisy)

    show_images(x_noisy, x_recon, x_clean)
