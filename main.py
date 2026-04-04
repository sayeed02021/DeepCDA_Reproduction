import os
import glob
import torch
from torch.utils.data import DataLoader
import argparse
import yaml


from dataset import getloaders
from utils import seed_everything, Logger
from train import train, validate
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
    os.makedirs(args.save_folder, exist_ok=True)

    for fold in range(args.n_folds):
        patience = 0
        best_val_loss = 1000
        logger = Logger(
        name=f'{args.dataset}_fold_{fold+1}',
        save_folder=args.save_folder
    )
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
            train_loss = train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                device=args.device,
                epoch=epoch+1
            )
            val_loss = validate(
                model=model, 
                loader=val_loader,
                criterion=torch.nn.MSELoss(),
                epoch=epoch, device=args.device
            )
            logger.update(fold+1, 'fold')
            logger.update(epoch+1, 'epoch')
            logger.update(train_loss, 'train_mse')
            logger.update(val_loss, 'val_mse')
            logger.plot(['train_mse', 'val_mse'])
            logger.save()

            if val_loss<=best_val_loss and epoch+1>=args.start_after:
                pt_file = glob.glob(f'{args.save_folder}/{args.dataset}_fold_{fold+1}_*.pt')
                for file in pt_file:
                    os.remove(file)
                
                model_save_path = f'{args.save_folder}/{args.dataset}_fold_{fold+1}_model_ep_{epoch+1}.pt'
                torch.save(model.state_dict(), model_save_path)
                best_val_loss = val_loss
                patience = 0

            elif epoch+1>=args.start_after and val_loss>=best_val_loss:
                patience+=1

            if patience>=args.patience:
                print("Val Loss saturation. Exit training for this fold")
                break




        # torch.save(model.state_dict(), f'model_fold_{fold+1}.pt')


if __name__=='__main__':
    main()




