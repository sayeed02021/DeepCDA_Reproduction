import torch
import torch.nn as nn
from tqdm import tqdm

from models import DeepCDA, Discriminator

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
    Train the discriminator while keeping test_encoder fixed
    
    """
    train_encoder.eval()
    test_encoder.eval()
    discriminator.train()

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(
        zip(train_loader, test_loader),
        desc=f'Discriminator Epoch: {epoch}',
        dynamic_ncols=True,
        leave=False,
        total = min(len(train_loader), len(test_loader))
    )

    total_loss = 0

    for idx, ((d_train, p_train, _), (d_test, p_test, _)) in enumerate(pbar):
        d_train, p_train = d_train.to(device), p_train.to(device)
        d_test, p_test = d_test.to(device), p_test.to(device)

        with torch.no_grad():
            F_train, _ = train_encoder(d_train, p_train)
            F_test, _ = test_encoder(d_test, p_test)

        train_targets = torch.ones(F_train.size(0), device=device)
        test_targets = torch.zeros(F_test.size(0), device=device)
        # disc_train = discriminator(F_train)
        # disc_test = discriminator(F_test)
        # print(disc_train, disc_test)
        disc_optimizer.zero_grad()
        train_out = discriminator(F_train)
        test_out = discriminator(F_test)
        
        loss = criterion(train_out, train_targets) + criterion(test_out, test_targets)
        loss.backward()
        disc_optimizer.step()

        total_loss+=loss.item()
        # print(torch.argmax(train_out, dim=-1), torch.argmax(test_out, dim=-1))
        pbar.set_postfix({
            "Disc Loss": f"{total_loss/(idx+1):0.3f}"
        })

    pbar.close()
    return total_loss/min(len(train_loader), len(test_loader))



def train_test_encoder(
        test_encoder: DeepCDA,
        discriminator: Discriminator,
        test_loader: torch.utils.data.DataLoader,
        enc_optimizer: torch.optim.Optimizer,
        epoch: int,
        device: str = 'cpu'

):
    """
    Train target encoder
    
    """
    test_encoder.train()
    discriminator.eval()

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(
        test_loader,
        desc=f"Target Enc Epoch {epoch}",
        dynamic_ncols=True,
        leave=False
    )

    total_loss = 0
    for idx, (d_test, p_test, _) in enumerate(pbar):
        d_test, p_test = d_test.to(device), p_test.to(device)

        enc_optimizer.zero_grad()

        F_test, _ = test_encoder(d_test, p_test)

        targets = torch.ones(F_test.size(0), device=device)
        loss = criterion(discriminator(F_test), targets)

        loss.backward()
        enc_optimizer.step()

        total_loss += loss.item()
        # print(loss.item())


        pbar.set_postfix({'G Loss': f'{total_loss / (idx + 1):.3f}'})


    pbar.close()
    return total_loss/(len(test_loader))