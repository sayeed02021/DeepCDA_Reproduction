import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import yaml

from dataset import getloaders
from train_test_utils import test
from models import DeepCDA
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


def test_and_compute_metrics(protein_k, smiles_k, out_dim, save_folder, args):
    model = DeepCDA(
        smiles_dict_len=64,
        protein_dict_len=25,
        embedding_size=args.embedding_size,
        out_dim = out_dim,
        protein_k=protein_k,
        smiles_k=smiles_k,
        method_type=args.method
    )
    train_loader, val_loader, test_loader = getloaders(args.test_dataset, 
                                                       0,
                                                       args.batch_size)
    all_targets, all_preds_fold_wise = test(
        loader = test_loader,
        model_folder = save_folder,
        model = model,
        device = args.device
    )
    averaged_preds = all_preds_fold_wise.mean(dim=0).squeeze()
    n_folds = all_preds_fold_wise.shape[0]
    Rm_fold_wise = np.zeros(n_folds)
    for fold in range(n_folds):
        a_pred = all_preds_fold_wise[fold]

        R,Rm,r2 = compute_metrics(all_targets, a_pred.squeeze())
        Rm_fold_wise[fold] = Rm
    
    mse = F.mse_loss(all_preds_fold_wise.mean(dim=0).squeeze(), all_targets)
    R,_,r2 = compute_metrics(all_targets, averaged_preds)
    print("mse, pearson, rm^2(mean and std), r2_score: ", mse, R, Rm_fold_wise.mean(), Rm_fold_wise.std(), r2)
    data = {}
    data['name'] = ['DAVIS_paper_code']
    data['Pearson'] = [R]
    data['MSE'] = [mse.item()]
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
    seed_everything(0)
    args = parse_args()
    args.method = 'paper'
    test_and_compute_metrics(
        protein_k=args.protein_k,
        smiles_k = args.smiles_k,
        out_dim = args.out_dim,
        save_folder = '../own_results/davis',
        args = args
    )

if __name__=='__main__':
    main()
    