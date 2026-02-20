# src/utils.py
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Fixe toutes les sources de hasard
    """
    random.seed(seed)        # Python
    np.random.seed(seed)     # NumPy
    torch.manual_seed(seed)  # PyTorch CPU


def add_noise(x, noise_std):
    """
    Ajoute un bruit gaussien à un tensor.
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise
