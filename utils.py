import torch
import numpy as np
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import pickle

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


def r_squared(y_true, y_pred):
    """Standard R² (with intercept)"""
    r, _ = stats.pearsonr(y_true, y_pred)
    return r ** 2

def r0_squared(y_true, y_pred):
    """R² with zero intercept (no bias term) via OLS through origin"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Slope through origin: beta = sum(y_true * y_pred) / sum(y_pred^2)
    beta = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    y_pred_0 = beta * y_pred
    ss_res = np.sum((y_true - y_pred_0) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def compute_metrics(y_true:torch.Tensor, y_pred:torch.Tensor):
    R = r_squared(y_true, y_pred)

    R0 = r0_squared(y_true, y_pred).item()

    Rm = R*(1-np.sqrt(R-R0))

    r2 = r2_score(y_true, y_pred)

    return R, Rm, r2

def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data





    