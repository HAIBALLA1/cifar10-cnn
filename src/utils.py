# src/utils.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_noise(x, noise_std=0.2):
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    return torch.clamp(x_noisy, -1.0, 1.0)