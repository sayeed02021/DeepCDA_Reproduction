import torch
from tqdm import tqdm
import torch.nn as nn
from models import DeepCDA
import glob
import copy
import random

def train(model, loader, optimizer, criterion, epoch, device, batch_fraction=1.0):
    model.train()
    
    # Randomly sample a fraction of batch indices
    total_batches = len(loader)
    num_batches = max(1, int(total_batches * batch_fraction))
    selected_indices = set(random.sample(range(total_batches), num_batches))
    
    pbar = tqdm(total=num_batches, desc=f'Train Epoch: {epoch}', dynamic_ncols=True, leave=False)
    total_loss = 0
    trained_batches = 0

    for idx, (d, p, a) in enumerate(loader):
        if idx not in selected_indices:
            continue
        
        d, p, a = d.to(device), p.to(device), a.to(device)
        optimizer.zero_grad()
        F, a_pred = model(
            drug_seq=d,
            protein_seq=p
        )
        loss = criterion(a_pred.squeeze(), a.squeeze())
        loss.backward()
        optimizer.step()

        trained_batches += 1
        total_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'Loss': f'{total_loss / trained_batches:.3f}'})

    pbar.close()
    return total_loss / trained_batches


def validate(model, loader, criterion, epoch, device, batch_fraction):
    model.eval()
    # Randomly sample a fraction of batch indices
    total_batches = len(loader)
    num_batches = max(1, int(total_batches * batch_fraction))

    selected_indices = set(random.sample(range(total_batches), num_batches))

    pbar = tqdm(total=num_batches, desc=f'Val Epoch: {epoch}', dynamic_ncols=True, leave=False)
    total_loss = 0

    tracked_batches=0
    with torch.no_grad():
        for idx, (d,p,a) in enumerate(loader):
            if idx not in selected_indices:
                continue
            d,p,a = d.to(device),p.to(device),a.to(device)
            
            F, a_pred = model(
                drug_seq = d,
                protein_seq = p
            )
            loss = criterion(a_pred.squeeze(), a.squeeze())
            
            total_loss +=loss.item()
            tracked_batches+=1
            pbar.update(1)
            pbar.set_postfix(
                {
                    'Loss': f'{total_loss/(tracked_batches+1):0.3f}'
                }
            )

        pbar.close()

    return total_loss/tracked_batches



def test(loader: torch.utils.data.DataLoader, 
         model_folder: str, 
         model: DeepCDA,
         device='cpu'):
    
    model_paths = glob.glob(model_folder+'/*.pt')
    if len(model_paths)==0:
        raise ValueError(f"No model weights found in {model_folder}")
    models = []
    for path in model_paths:
        m = copy.deepcopy(model)                                    # fresh copy
        m.load_state_dict(torch.load(path, map_location='cpu'))     # load weights in-place
        models.append(m) 

    pbar = tqdm(loader, desc=f'Test', dynamic_ncols=True, leave=False)
    all_targets = []
    all_preds_fold_wise = []
    with torch.no_grad():
        for idx, (d,p,a) in enumerate(pbar):
            d,p,a = d.to(device),p.to(device),a.to(device)
            
            a_folds = []
            for m in models:
                m.to(device)
                m.eval()

                F, a_pred = m(
                    drug_seq = d,
                    protein_seq = p
                )
                a_folds.append(a_pred)
            a_fold = torch.stack(a_folds, dim=0) # (n_fold, batch_size, 1)


            
            all_targets.append(a.cpu())
            all_preds_fold_wise.append(a_fold.cpu())
        pbar.close()

    all_targets = torch.cat(all_targets, dim=0)
    all_preds_fold_wise = torch.cat(all_preds_fold_wise, dim=1) # (n_fold, total_examples, 1)

    return all_targets, all_preds_fold_wise



        
