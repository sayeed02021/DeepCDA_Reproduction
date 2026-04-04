import torch
import numpy as np
import random

def seed_everything(seed: int):
    """
    Set seed for reproducibility across:
    - Python random
    - NumPy
    - PyTorch (CPU + CUDA)s
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic CUDA ops (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False