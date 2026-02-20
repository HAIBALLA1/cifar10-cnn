# scripts/benchmark.py
# Robustness benchmark for the Denoising Autoencoder (DAE)


import os
import math
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
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


def denormalize_to_01(x: torch.Tensor) -> torch.Tensor:

    mean = MEAN_T.to(x.device)
    std = STD_T.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    """
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def evaluate_metrics(model: torch.nn.Module, loader, device: torch.device, noise_std: float) -> Tuple[float, float]:

    model.eval()

    total_mse = 0.0
    total_count = 0

    for x_clean, _ in loader:
        x_clean = x_clean.to(device, non_blocking=True)
        x_noisy = add_noise(x_clean, noise_std)
        x_recon = model(x_noisy)

        x_clean_01 = denormalize_to_01(x_clean)
        x_recon_01 = denormalize_to_01(x_recon)

        mse = F.mse_loss(x_recon_01, x_clean_01, reduction="mean").item()

        bs = x_clean.size(0)
        total_mse += mse * bs
        total_count += bs

    mse_avg = total_mse / max(total_count, 1)
    psnr = psnr_from_mse(mse_avg, max_val=1.0)
    return mse_avg, psnr


@torch.no_grad()
def save_recon_grid_png(model: torch.nn.Module, loader, device: torch.device, noise_std: float, out_path: str, n: int = 8):

    model.eval()
    x_clean, _ = next(iter(loader))
    x_clean = x_clean.to(device, non_blocking=True)
    x_noisy = add_noise(x_clean, noise_std)
    x_recon = model(x_noisy)

    # Move to [0,1] for visualization
    x_clean = denormalize_to_01(x_clean[:n]).cpu()
    x_noisy = denormalize_to_01(x_noisy[:n]).cpu()
    x_recon = denormalize_to_01(x_recon[:n]).cpu()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(n * 2.2, 6))

    def _plot_row(tensor, row_idx, title):
        for i in range(n):
            ax = plt.subplot(3, n, row_idx * n + i + 1)
            ax.imshow(tensor[i].permute(1, 2, 0))
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(title)

    _plot_row(x_noisy, 0, "Noisy")
    _plot_row(x_recon, 1, "Recon")
    _plot_row(x_clean, 2, "Clean")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_csv(rows: List[Tuple[float, float, float]], out_path: str):

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_std", "test_mse", "psnr_db"])
        for ns, mse, psnr in rows:
            w.writerow([ns, mse, psnr])


def smoke_checks(model: torch.nn.Module, device: torch.device):
    x = torch.randn(2, 3, 32, 32, device=device)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN/Inf"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if "benchmark" not in cfg:
        cfg.benchmark = {}
    if "noise_levels" not in cfg.benchmark:
        # default list
        cfg.benchmark.noise_levels = [0.05, 0.10, 0.20, 0.30]
    if "viz_noise_std" not in cfg.benchmark:
        cfg.benchmark.viz_noise_std = 0.20

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    if "data" in cfg and "use_augmentation" in cfg.data:
        cfg.data.use_augmentation = False

    _, test_loader = get_loaders(cfg)

    model = build_model(cfg).to(device)
    smoke_checks(model, device)

    ckpt_path = Path(cfg.train.save_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable: {ckpt_path}\n"
            f"Lance d'abord l'entraînement ou override: train.save_path=dae_best.pth"
        )

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- MLflow setup
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = getattr(cfg, "mlflow", {}).get("experiment_name", None) if hasattr(cfg, "mlflow") else None
    if not exp_name:
        exp_name = "cifar10-dae"
    mlflow.set_experiment(exp_name)

    run_name = f"benchmark_{ckpt_path.stem}"

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = out_dir / "config_benchmark.yaml"
    cfg_file.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    noise_levels = list(cfg.benchmark.noise_levels)
    rows = []

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("checkpoint", str(ckpt_path))
        mlflow.log_param("device", str(cfg.train.device))
        mlflow.log_param("noise_levels", str(noise_levels))
        mlflow.log_param("viz_noise_std", float(cfg.benchmark.viz_noise_std))

        mlflow.log_artifact(str(cfg_file), artifact_path="config")

        print("\n=== Robustness Benchmark (test set) ===")
        for ns in noise_levels:
            mse, psnr = evaluate_metrics(model, test_loader, device, float(ns))
            rows.append((float(ns), float(mse), float(psnr)))

            mlflow.log_metric("test_mse", float(mse), step=int(round(float(ns) * 1000)))
            mlflow.log_metric("psnr_db", float(psnr), step=int(round(float(ns) * 1000)))

            mlflow.log_metric(f"mse_noise_{ns}", float(mse))
            mlflow.log_metric(f"psnr_noise_{ns}", float(psnr))

            print(f"noise_std={ns:.2f} | MSE={mse:.6f} | PSNR={psnr:.2f} dB")

        #  CSV table
        csv_path = out_dir / "robustness_table.csv"
        write_csv(rows, str(csv_path))
        mlflow.log_artifact(str(csv_path), artifact_path="tables")

        grid_path = out_dir / "recon_grid_benchmark.png"
        save_recon_grid_png(model, test_loader, device, float(cfg.benchmark.viz_noise_std), str(grid_path), n=8)
        mlflow.log_artifact(str(grid_path), artifact_path="plots")

        print(f"\n[Saved] {csv_path}")
        print(f"[Saved] {grid_path}")
        print(f"[MLflow] experiment='{exp_name}', run='{run_name}'")


if __name__ == "__main__":
    main()