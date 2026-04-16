import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torch.nn.functional as F
from scipy.spatial import distance
from models import DeepCDA, Discriminator

def compute_sample_weights(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between source and target features

    :param source_feat: (N,D) 
    :param target_feat: (N,D)
    :return (2B,) sample weights for source and target concatenated
    """

    source_feat_np = source_feat.detach().cpu().numpy()
    target_feat_np = target_feat.detach().cpu().numpy()

    ddist = distance.cdist(source_feat_np, target_feat_np, 'cosine') # (N,N)

    N = source_feat_np.shape[0]

    tmp_samp = np.amax(np.eye(N)-(1-ddist), axis=1) # (N,) get the samples with maximum source cosine similarity
    tmp_samp = tmp_samp[:,None] # (N,1)
    source_weights = np.exp(tmp_samp) / np.sum(np.exp(tmp_samp))  # (N, 1)

    # uniform target weigths
    lbl_source = np.ones((N, 1))
    target_weights = np.exp(lbl_source) / np.sum(np.exp(lbl_source))  # (N, 1)
    sample_weights = np.concatenate([source_weights, target_weights], axis=0)  # (2N, 1)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32, device=source_feat.device).squeeze() # (2N,) 
    return sample_weights


def apply_label_noise_and_soft_labels(labels: np.ndarray) -> np.ndarray:
    """
    Apply label noise and soft labels.

    :param labels: (2N,) binary labels
    :return: (2N,) noisy soft labels
    """
    labels = labels.copy().astype(np.float32)
    total = len(labels)

    # Flip labels
    noise_idx = np.random.choice(total, int(np.floor(0.05 * total)), replace=False)
    for idx in noise_idx:
        labels[idx] = 1.0 if labels[idx] == 0.0 else 0.0

    # Soft labels
    soft_idx = np.random.choice(total, int(np.floor(0.5 * total)), replace=False)
    for idx in soft_idx:
        labels[idx] = 0.1 if labels[idx] == 0.0 else 0.9

    return labels

def train_discriminator(
        train_encoder: DeepCDA,
        test_encoder: DeepCDA,
        discriminator: Discriminator,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        disc_optimizer: torch.optim.Optimizer,
        epoch: int,
        device: str = 'cpu'
):
    """
    Train the discriminator while keeping both encoders fixed.
    """
    train_encoder.eval()
    test_encoder.eval()
    discriminator.train()


    criterion = nn.BCEWithLogitsLoss(reduction='none')

    pbar = tqdm(
        zip(train_loader, test_loader),
        desc=f'Discriminator Epoch: {epoch}',
        dynamic_ncols=True,
        leave=False,
        total=min(len(train_loader), len(test_loader))
    )

    total_loss = 0
    for idx, ((d_train, p_train, _), (d_test, p_test, _)) in enumerate(pbar):
        d_train, p_train = d_train.to(device), p_train.to(device)
        d_test, p_test = d_test.to(device), p_test.to(device)

        with torch.no_grad():
            F_train, _ = train_encoder(d_train, p_train)  # (N, D)
            F_test, _ = test_encoder(d_test, p_test)      # (N, D)


        sample_weights = compute_sample_weights(F_train, F_test)

        N = F_train.size(0)

        labels_np = np.concatenate([
            np.ones(N, dtype=np.float32),
            np.zeros(N, dtype=np.float32)
        ])

        # label noise and soft labels
        labels_np = apply_label_noise_and_soft_labels(labels_np)  # (2N,)
        labels = torch.tensor(labels_np, dtype=torch.float32, device=device)  # (2N,)

        
        feat_all = torch.cat([F_train, F_test], dim=0)  # (2N, D)

        disc_optimizer.zero_grad()

        out = discriminator(feat_all).squeeze(-1)  # (2N,)

        loss = (criterion(out, labels) * sample_weights).mean()
        loss.backward()
        disc_optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Disc Loss": f"{total_loss / (idx + 1):0.3f}"})

    pbar.close()
    return total_loss / min(len(train_loader), len(test_loader))



def train_test_encoder(
        test_encoder: DeepCDA,
        discriminator: Discriminator,
        test_loader: torch.utils.data.DataLoader,
        enc_optimizer: torch.optim.Optimizer,
        epoch: int,
        device: str = 'cpu',
        batch_ratio: float = 1.0
):
    """
    Train target encoder while keeping discriminator fixed.
    """
    test_encoder.train()
    discriminator.eval()

    
    criterion = nn.BCEWithLogitsLoss()

    total_batches = len(test_loader)
    num_batches = max(1, int(total_batches * batch_ratio))
    selected_indices = set(random.sample(range(total_batches), num_batches))

    pbar = tqdm(
        total=num_batches,
        desc=f"Target Enc Epoch {epoch}",
        dynamic_ncols=True,
        leave=False
    )

    trained_batches = 0
    total_loss = 0

    for idx, (d_test, p_test, _) in enumerate(test_loader):
        if idx not in selected_indices:
            continue

        d_test, p_test = d_test.to(device), p_test.to(device)
        enc_optimizer.zero_grad()

        F_test, _ = test_encoder(d_test, p_test)  # (N, D)

        
        targets = torch.ones(F_test.size(0), device=device)  # (N,)
        out = discriminator(F_test).squeeze(-1)               # (N,)
        loss = criterion(out, targets)
        loss.backward()
        enc_optimizer.step()

        trained_batches += 1
        total_loss += loss.item()

        pbar.update(1)
        pbar.set_postfix({'Encoder Loss': f'{total_loss / trained_batches:.3f}'})

    pbar.close()
    return total_loss / num_batches