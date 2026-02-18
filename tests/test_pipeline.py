import torch
from src.model import DenoisingAutoencoder
from src.data import get_cifar10_loaders
from src.utils import add_noise

def test_model_output_shape():
    model = DenoisingAutoencoder()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == x.shape

def test_dataloader_shape():
    train_loader, _ = get_cifar10_loaders(batch_size=4, use_augmentation=False)
    x, _ = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32)

def test_noise_shape():
    x = torch.randn(2, 3, 32, 32)
    y = add_noise(x, 0.2)
    assert y.shape == x.shape
