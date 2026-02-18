from src.data import get_cifar10_loaders
from src.model import DenoisingAutoencoder
from src.train import train
from src.utils import get_device, set_seed

set_seed(42)
device = get_device()

train_loader, test_loader = get_cifar10_loaders(batch_size=128, use_augmentation=True)
model = DenoisingAutoencoder()

train(model, train_loader, test_loader, device, epochs=30, lr=1e-3, noise_std=0.2)
