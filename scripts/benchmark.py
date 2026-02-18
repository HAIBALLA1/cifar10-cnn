import os
import math
import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.data import get_cifar10_loaders
from src.model import DenoisingAutoencoder
from src.utils import get_device, set_seed, add_noise


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    # PSNR = 10 * log10(MAX^2 / MSE)
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def evaluate_metrics(model, loader, device, noise_std: float):
    model.eval()
    criterion = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    n_batches = 0

    for x_clean, _ in loader:
        x_clean = x_clean.to(device)
        x_noisy = add_noise(x_clean, noise_std)
        x_recon = model(x_noisy)

        loss = criterion(x_recon, x_clean).item()
        total_loss += loss
        n_batches += 1

    mse = total_loss / max(n_batches, 1)
    # Attention: tes images sont normalisées (pas dans [0,1])
    # PSNR reste utile comme comparaison relative. (Sinon il faut dénormaliser)
    psnr = psnr_from_mse(mse, max_val=1.0)
    return mse, psnr


def smoke_checks(model, device):
    # Check shapes
    x = torch.randn(2, 3, 32, 32).to(device)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Check NaNs
    assert torch.isfinite(y).all(), "Output contains NaN/Inf"


def save_recon_grid(model, loader, device, noise_std: float, out_path: str):
    model.eval()
    x_clean, _ = next(iter(loader))
    x_clean = x_clean.to(device)
    x_noisy = add_noise(x_clean, noise_std)

    with torch.no_grad():
        x_recon = model(x_noisy)

    # Sauvegarde 3 lignes: noisy / recon / clean
    # (On sauvegarde "tel quel" dans l’espace normalisé)
    grid = torch.cat([x_noisy[:8], x_recon[:8], x_clean[:8]], dim=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path, nrow=8, normalize=True)
    print(f"Saved recon grid to: {out_path}")


def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128, use_augmentation=False)

    model = DenoisingAutoencoder().to(device)
    smoke_checks(model, device)
    print("✅ Smoke checks passed")

    # Charger un modèle si dispo
    ckpt = "dae_best.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("⚠️ No checkpoint found (dae_best.pth). Metrics will be for an untrained model.")

    # Tester plusieurs niveaux de bruit
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    print("\n=== Benchmark on test set ===")
    for ns in noise_levels:
        mse, psnr = evaluate_metrics(model, test_loader, device, ns)
        print(f"noise_std={ns:.2f} | MSE={mse:.6f} | PSNR~={psnr:.2f} dB")

    # Sauver une grille d’images
    save_recon_grid(model, test_loader, device, noise_std=0.2, out_path="images/recon_grid.png")


if __name__ == "__main__":
    main()
