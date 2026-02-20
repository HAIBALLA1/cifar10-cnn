import hydra
import mlflow
from contextlib import nullcontext
from omegaconf import DictConfig, OmegaConf

from src.data import get_loaders
from src.model import build_model
from src.train import train
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.train.seed)

    # MLflow setup
    if cfg.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    cm = mlflow.start_run(run_name=cfg.mlflow.run_name) if cfg.mlflow.enabled else nullcontext()

    with cm:
        if cfg.mlflow.enabled:

            mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

            mlflow.log_params({
                # train
                "epochs": int(cfg.train.epochs),
                "lr": float(cfg.train.lr),
                "device": str(cfg.train.device),
                "seed": int(cfg.train.seed),
                "noise_std": float(cfg.train.noise_std),
                "scheduler_step_size": int(cfg.train.scheduler_step_size),
                "scheduler_gamma": float(cfg.train.scheduler_gamma),
                "save_path": str(cfg.train.save_path),

                # data
                "data_dir": str(cfg.data.data_dir),
                "batch_size": int(cfg.data.batch_size),
                "num_workers": int(cfg.data.num_workers),
                "use_augmentation": bool(cfg.data.use_augmentation),

                # model
                "model_name": str(cfg.model.name),
                "in_channels": int(cfg.model.in_channels),
                "c1": int(cfg.model.c1),
                "c2": int(cfg.model.c2),
            })

        # Pipeline
        train_loader, test_loader = get_loaders(cfg)
        model = build_model(cfg)

        # Train retourne un dict de métriques
        metrics = train(model, train_loader, test_loader, cfg)

        if cfg.mlflow.enabled and isinstance(metrics, dict):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()