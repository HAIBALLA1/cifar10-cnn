import hydra
from omegaconf import DictConfig

from src.data import get_loaders
from src.model import build_model
from src.train import train
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.train.seed)

    train_loader, test_loader = get_loaders(cfg)
    model = build_model(cfg)

    train(model, train_loader, test_loader, cfg)


if __name__ == "__main__":
    main()
