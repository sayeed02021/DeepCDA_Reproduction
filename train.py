import torch
from tqdm import tqdm
from models import DeepCDA
import torch.nn as nn

def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    pbar = tqdm(loader, desc=f'Epoch: {epoch}', dynamic_ncols=True, leave=False)
    total_loss = 0
    for idx, (d,p,a) in enumerate(pbar):
        d,p,a = d.to(device),p.to(device),a.to(device)
        optimizer.zero_grad()
        a_pred = model(
            drug_seq = d,
            protein_seq = p
        )
        loss = criterion(a_pred.squeeze(), a)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()

        pbar.set_postfix(
            {
                'Loss': f'{total_loss/(idx+1):0.3f}'
            }
        )

    pbar.close()



        
