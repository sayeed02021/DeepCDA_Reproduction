import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import yaml
import copy
from natsort import natsorted

from models import DeepCDA, Discriminator
from ada_train_utils import train_discriminator, train_test_encoder
from dataset import getloaders
from utils import seed_everything, Logger, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to train/val/test configs")
    config_args = parser.parse_args()
    config_path = config_args.config_path

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    args = argparse.Namespace(**config_data)

    return args


def load_pre_trained_model(model: DeepCDA, model_path:str, device:str):
    
    model_wghts = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_wghts)

    test_encoder = copy.deepcopy(model)
    return model.to(device), test_encoder.to(device)


def train_encoder_and_discriminator(
        train_encoder: DeepCDA,
        test_encoder: DeepCDA,
        discriminator: Discriminator,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        enc_optmizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        epoch: int,
        device:str='cpu'
):
    disc_loss = train_discriminator(
        train_encoder=train_encoder,
        test_encoder=test_encoder,
        discriminator=discriminator,
        train_loader=train_loader,
        test_loader=test_loader,
        disc_optimizer=disc_optimizer,
        epoch=epoch,
        device=device
    )

    enc_loss = train_test_encoder(
        test_encoder=test_encoder,
        discriminator=discriminator, 
        test_loader=test_loader,
        enc_optimizer=enc_optmizer,
        epoch=epoch,
        device=device
    )

    return disc_loss, enc_loss

    


def main():
    seed_everything(0)
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    model_paths = natsorted(glob.glob(args.saved_model_folder+"/*.pt"))

    for fold in range(args.n_folds):
        print("Fold: ", fold+1)
        fold_data = {}
        train_loader, _, _ = getloaders(args.train_dataset, fold,
                                        args.batch_size)
        test_loader, _, _ = getloaders(args.test_dataset, fold,
                                        args.batch_size)
        


        
        
        model = DeepCDA(
            smiles_dict_len=64,
            protein_dict_len=25,
            embedding_size=args.embedding_size,
            out_dim = args.out_dim,
            protein_k=args.protein_k,
            smiles_k = args.smiles_k,
            method_type = args.method
        )

        train_encoder, test_encoder = load_pre_trained_model(model, model_paths[fold], device=args.device)
        all_enc_loss = []
        all_disc_loss = []
        discriminator = Discriminator(6*args.out_dim).to(args.device)
        enc_optmizer=torch.optim.Adam(test_encoder.parameters(), lr=args.lr_enc)
        disc_optimizer=torch.optim.Adam(discriminator.parameters(), lr = args.warmup_lr)

        # pre-train discriminator
        print("Pre-training discriminator")
        for epoch in range(args.warmup_epochs):
            disc_loss = train_discriminator(
                train_encoder=train_encoder,
                test_encoder=test_encoder,
                discriminator=discriminator,
                train_loader=train_loader,
                test_loader=test_loader,
                disc_optimizer=disc_optimizer,
                epoch=epoch+1,
                device=args.device
            )
        disc_optimizer=torch.optim.Adam(discriminator.parameters(), lr = args.lr_disc)
        print('Starting adversarial training')
        for epoch in range(args.n_epochs):
            disc_loss, enc_loss = train_encoder_and_discriminator(
                train_encoder=train_encoder,
                test_encoder=test_encoder,
                discriminator=discriminator,
                train_loader=train_loader,
                test_loader=test_loader,
                enc_optmizer=enc_optmizer,
                disc_optimizer=disc_optimizer,
                device = args.device,
                epoch=epoch+1

            )

            all_enc_loss.append(enc_loss)
            all_disc_loss.append(disc_loss)

        fold_data['Enc_Loss'] = all_enc_loss
        fold_data["Disc_Loss"] = all_disc_loss
        df = pd.DataFrame(fold_data)
        df.to_csv(f'{args.save_folder}/fold_{fold+1}.csv', index=False)
        model_path = f'{args.save_folder}/model_fold_{fold+1}.pt'

        torch.save(test_encoder.state_dict(), model_path)


if __name__=="__main__":
    main()


    

        
        

    
