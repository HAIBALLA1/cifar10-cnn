# src/eval.py
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from src.data import get_loaders
from src.model import build_model
from src.utils import add_noise, set_seed


MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T  = torch.tensor(STD).view(1, 3, 1, 1)


def unnormalize(x):
    mean = MEAN_T.to(x.device)
    std = STD_T.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def show_images(x_noisy, x_recon, x_clean, n=5):
    x_noisy = unnormalize(x_noisy[:n].cpu())
    x_recon = unnormalize(x_recon[:n].cpu())
    x_clean = unnormalize(x_clean[:n].cpu())

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):

        plt.subplot(3, n, i + 1)
        plt.imshow(x_noisy[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy")


        plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_recon[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed")

        # Clean
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(x_clean[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Clean")

    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):


    set_seed(cfg.train.seed)


    cfg.data.use_augmentation = False


    device = torch.device(cfg.train.device)

    _, test_loader = get_loaders(cfg)

    # build model via cfg.model
    model = build_model(cfg).to(device)

    # load weights
    model_path = cfg.train.save_path
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # un batch
    x_clean, _ = next(iter(test_loader))
    x_clean = x_clean.to(device)

    # bruit depuis config
    x_noisy = add_noise(x_clean, cfg.train.noise_std)

    with torch.no_grad():
        x_recon = model(x_noisy)

    show_images(x_noisy, x_recon, x_clean)


if __name__ == "__main__":
    main()