import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import argparse
import yaml


from dataset import getloaders
from utils import seed_everything, Logger, compute_metrics, load_pkl
from train_test_utils import train, validate, test
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

def train_per_setting(fold, protein_k, smiles_k, lr, out_dim, save_folder, args):
    patience = 0
    best_val_loss = 1000
    logger = Logger(
    name=f'{args.dataset}_fold_{fold+1}',
    save_folder=save_folder
)
    print(f'FOLD: {fold+1}')
    train_loader, val_loader, test_loader = getloaders(args.dataset, 
                                                    fold, 
                                                    args.batch_size)

    
    model = DeepCDA(
        smiles_dict_len=64,
        protein_dict_len=25,
        embedding_size=args.embedding_size,
        out_dim = out_dim,
        protein_k=protein_k,
        smiles_k=smiles_k,
        method_type=args.method
    )

    
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(args.n_epochs):
        train_loss = train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=args.device,
            epoch=epoch+1,
            batch_fraction=args.batch_ratio
        )
        val_loss = validate(
            model=model, 
            loader=val_loader,
            criterion=torch.nn.MSELoss(),
            epoch=epoch, device=args.device,
            batch_fraction=args.batch_ratio
        )
        logger.update(fold+1, 'fold')
        logger.update(epoch+1, 'epoch')
        logger.update(train_loss, 'train_mse')
        logger.update(val_loss, 'val_mse')
        logger.plot(['train_mse', 'val_mse'])
        logger.save()

        if val_loss<=best_val_loss and epoch+1>=args.start_after:
            pt_file = glob.glob(f'{save_folder}/{args.dataset}_fold_{fold+1}_*.pt')
            for file in pt_file:
                os.remove(file)
            
            model_save_path = f'{save_folder}/{args.dataset}_fold_{fold+1}_model_ep_{epoch+1}.pt'
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = val_loss
            patience = 0

        elif epoch+1>=args.start_after and val_loss>=best_val_loss:
            patience+=1

        if patience>=args.patience:
            print(f"Val Loss saturation reached in epoch {epoch+1}, Exiting training for this fold")
            break


def test_and_compute_metrics(protein_k, smiles_k, out_dim, save_folder, args, scaler):
    model = DeepCDA(
        smiles_dict_len=64,
        protein_dict_len=25,
        embedding_size=args.embedding_size,
        out_dim = out_dim,
        protein_k=protein_k,
        smiles_k=smiles_k,
        method_type=args.method
    )
    train_loader, val_loader, test_loader = getloaders(args.dataset, 
                                                       0,
                                                       args.batch_size)
    all_targets, all_preds_fold_wise = test(
        loader = test_loader,
        model_folder = save_folder,
        model = model,
        device = args.device
    )
    averaged_preds = all_preds_fold_wise.mean(dim=0)
    n_folds = all_preds_fold_wise.shape[0]
    Rm_fold_wise = np.zeros(n_folds)
    for fold in range(n_folds):
        a_pred = all_preds_fold_wise[fold]

        R,Rm,r2 = compute_metrics(scaler.inverse_transform(all_targets), scaler.inverse_transform(a_pred))
        
        Rm_fold_wise[fold] = Rm[0]
    all_preds_unscaled = torch.tensor(scaler.inverse_transform(all_preds_fold_wise.mean(dim=0))).squeeze()
    all_targets_unscaled = torch.tensor(scaler.inverse_transform(all_targets)).squeeze()
    
    print(type(all_preds_unscaled))
    print(type(all_targets_unscaled))
    print(type(all_preds_unscaled.size))    # should say <built-in method size ...>
    print(type(all_targets_unscaled.size))  # if this prints <class 'int'>, that's your culprit
    print(all_preds_unscaled.shape)
    print(all_targets_unscaled.shape)
    mse = F.mse_loss(all_preds_unscaled, all_targets_unscaled).item()
    R,_,r2 = compute_metrics(all_preds_unscaled, all_targets_unscaled)
    print("mse, pearson, rm^2(mean and std), r2_score: ", mse, R, Rm_fold_wise.mean(), Rm_fold_wise.std(), r2)
    data = {}
    data['name'] = ['DAVIS_github_small_model']
    data['Pearson'] = [R]
    data['MSE'] = [mse]
    data['r2'] = r2
    data['rm_sq_mean'] = [Rm_fold_wise.mean()]
    data['rm_sq_std'] = [Rm_fold_wise.std()]
    df = pd.DataFrame(data)
    if os.path.exists('metrics.csv'):
        original_data = pd.read_csv('metrics.csv')
        original_data = pd.concat([original_data, df])
        original_data.to_csv('metrics.csv', index=False)
    else:
        df.to_csv('metrics.csv', index=False)




def main():
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    total_settings = len(args.protein_k)*len(args.smiles_k)*len(args.out_dim)*len(args.lr)
    print("Total Settings: ", total_settings)
    setting_number = 0
    if args.train:
        for protein_k in args.protein_k:
            for smiles_k in args.smiles_k:
                for out_dim in args.out_dim:
                    for lr in args.lr:
                        setting_number+=1
                        save_folder = f'{args.save_folder}/SETTING_{setting_number}'
                        os.makedirs(save_folder, exist_ok=True)
                        setting_data = {}
                        setting_data['Setting'] = ['protein_filter', 'smiles_filter', 'num_filters', 'lr']
                        setting_data['Values'] = [protein_k, smiles_k, out_dim, lr]
                        setting_df = pd.DataFrame(setting_data)
                        setting_df.to_csv(f'{save_folder}/settings.txt', sep=" ", index=False)
                        for fold in range(args.n_folds):
                            train_per_setting(
                                fold=fold, 
                                protein_k=protein_k,
                                smiles_k=smiles_k,
                                lr=lr,
                                out_dim=out_dim,
                                save_folder=save_folder,
                                args=args
                            )
    
    if args.test:
        scaler = load_pkl(f'../Data_folded/{args.dataset}_scaler.pkl')
        i=0
        for protein_k in args.protein_k:
            for smiles_k in args.smiles_k:
                for out_dim in args.out_dim:
                    for lr in args.lr:  
                        # save_folder = args.save_folder+f'/SETTING_{i+1}' # change this if using anyother folder like args.save_folder + "SETTING_1" etc
                        save_folder = args.save_folder 
                        i+=1
                        test_and_compute_metrics(
                            protein_k=protein_k,
                            smiles_k=smiles_k,
                            out_dim=out_dim,
                            save_folder=save_folder,
                            args=args,
                            scaler=scaler
                        )      

if __name__=='__main__':
    main()




