from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
]

def get_transforms(use_augmentation=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if use_augmentation:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf


def get_loaders(cfg):
    """Hydra entrypoint: tout vient de cfg.data"""
    dcfg = cfg.data

    train_tf, test_tf = get_transforms(dcfg.use_augmentation)

    train_ds = datasets.CIFAR10(
        root=dcfg.data_dir,
        train=True,
        download=True,
        transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=dcfg.data_dir,
        train=False,
        download=True,
        transform=test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=dcfg.batch_size,
        shuffle=True,
        num_workers=dcfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=dcfg.batch_size,
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
