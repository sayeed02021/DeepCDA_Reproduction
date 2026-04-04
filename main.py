import torch
from torch.utils.data import DataLoader
import argparse
import yaml


from dataset import getloaders
from utils import seed_everything
from train import train
from models import DeepCDA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to train/val/test configs")
    config_args = parser.parse_args()
    config_path = config_args.config_path

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    args = argparse.Namespace(**config_data)
    return args


def main():
    args = parse_args()

    for fold in range(args.n_folds):
        print(f'FOLD: {fold+1}')
        train_loader, val_loader, test_loader = getloaders(args.dataset, 
                                                           fold, 
                                                           args.batch_size)

        model = DeepCDA(
            smiles_dict_len=64,
            protein_dict_len=25,
            embedding_size=args.embedding_size,
            out_dim = args.out_dim,
            protein_k=args.protein_k,
            smiles_k=args.smiles_k
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        print("Number of model parameters: ", n_params)

        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.n_epochs):
            train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                device=args.device,
                epoch=epoch+1
            )

        torch.save(model.state_dict(), f'model_fold_{fold+1}.pt')


if __name__=='__main__':
    main()




