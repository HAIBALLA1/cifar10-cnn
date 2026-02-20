# src/eval.py

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

import mlflow

from src.data import get_loaders
from src.model import build_model
from src.utils import add_noise, set_seed


MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T  = torch.tensor(STD).view(1, 3, 1, 1)


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = MEAN_T.to(x.device)
    std = STD_T.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def save_recon_grid(x_noisy: torch.Tensor, x_recon: torch.Tensor, x_clean: torch.Tensor, out_path: str, n: int = 6):
    """
    Sauvegarde une grille (Noisy / Reconstructed / Clean) en PNG.
    """
    x_noisy = unnormalize(x_noisy[:n].detach().cpu())
    x_recon = unnormalize(x_recon[:n].detach().cpu())
    x_clean = unnormalize(x_clean[:n].detach().cpu())

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):
        # Noisy
        plt.subplot(3, n, i + 1)
        plt.imshow(x_noisy[i].permute(1, 2, 0))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy")

        # Reconstructed
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def compute_test_mse(model: torch.nn.Module, test_loader, device: torch.device, noise_std: float) -> float:
    """
    Calcule un MSE moyen sur TOUT le test set.
    """
    model.eval()
    total_mse = 0.0
    total_count = 0

    for x_clean, _ in test_loader:
        x_clean = x_clean.to(device, non_blocking=True)
        x_noisy = add_noise(x_clean, noise_std)
        x_recon = model(x_noisy)

        # MSE moyen par batch (moyenne sur tous les pixels/canaux)
        mse = F.mse_loss(x_recon, x_clean, reduction="mean").item()

        # pondération par taille de batch (pour une moyenne globale correcte)
        bs = x_clean.size(0)
        total_mse += mse * bs
        total_count += bs

    return total_mse / max(total_count, 1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # --- seed + device ---
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    if "data" in cfg and "use_augmentation" in cfg.data:
        cfg.data.use_augmentation = False

    # --- data ---
    _, test_loader = get_loaders(cfg)

    # --- model ---
    model = build_model(cfg).to(device)

    model_path = Path(cfg.train.save_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {model_path}")

    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- MLflow setup ---

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = getattr(cfg, "mlflow", {}).get("experiment_name", None) if hasattr(cfg, "mlflow") else None
    if not experiment_name:
        # fallback: un nom lisible
        experiment_name = f"cifar10-eval-{getattr(cfg.model, 'name', 'model')}"
    mlflow.set_experiment(experiment_name)

    run_name = f"eval_{model_path.stem}"


    with mlflow.start_run(run_name=run_name):

        cfg_yaml = OmegaConf.to_yaml(cfg)
        hydra_out = Path("artifacts")
        hydra_out.mkdir(parents=True, exist_ok=True)
        cfg_file = hydra_out / "config_eval.yaml"
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        mlflow.log_artifact(str(cfg_file), artifact_path="config")

        mlflow.log_param("device", str(cfg.train.device))
        mlflow.log_param("noise_std", float(cfg.train.noise_std))
        mlflow.log_param("checkpoint", str(model_path))

        # Metric globale sur tout le test set
        test_mse = compute_test_mse(model, test_loader, device, float(cfg.train.noise_std))
        mlflow.log_metric("test_mse", float(test_mse))

        # Un batch pour visu + artifact image
        x_clean, _ = next(iter(test_loader))
        x_clean = x_clean.to(device, non_blocking=True)
        x_noisy = add_noise(x_clean, float(cfg.train.noise_std))
        x_recon = model(x_noisy)

        img_path = hydra_out / "recon_grid.png"
        save_recon_grid(x_noisy, x_recon, x_clean, str(img_path), n=6)
        mlflow.log_artifact(str(img_path), artifact_path="plots")


        mlflow.log_artifact(str(model_path), artifact_path="checkpoints")

        print(f"[EVAL] test_mse={test_mse:.6f}")
        print(f"[MLflow] experiment='{experiment_name}', run='{run_name}'")
        print(f"[Artifacts] {img_path}")

if __name__ == "__main__":
    main()