import torch
import numpy as np
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

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


class Logger:
    def __init__(self, name, save_folder):
        self.name = name
        self.save_folder = save_folder
        self.tracker = defaultdict(list)

    def update(self, val, key):
        self.tracker[key].append(val)
    
    def plot(self, keys):
        for key in keys:
            if key not in self.tracker.keys():
                print(f"{key} not being tracked")
                continue

            ep = np.arange(len(self.tracker[key]))
            plt.plot(ep, self.tracker[key], label=key)

        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.save_folder}/{self.name}_plot.png')
        plt.close()
        
        
        
    def save(self):
        df = pd.DataFrame(self.tracker)
        df.to_csv(f"{self.save_folder}/{self.name}_metrics.csv")

    