import torch
from torch.utils.data import Dataset, DataLoader
import scipy
import numpy as np
## dataset util functions

def load_mat_file(dataset_name: str, fold:int):
    if dataset_name.lower()=="davis":
        path = '../Data_folded/Davis/Davis_Dataset_folded.mat'
    elif dataset_name.lower()=='kiba':
        path = '../Data_folded/KIBA/KIBA_Dataset_folded.mat'
    
    data = scipy.io.loadmat(path) 
    train_drugs = data['train_folds_drugs'][fold] # n_fold, n, compound_seq_len
    train_proteins = data['train_folds_proteins'][fold] # n_fold, n, protein_seq_len
    train_y = data['train_folds_affinity'][fold] # n_fold, n 

    val_drugs = data['val_folds_drugs'][fold] # n_fold, n, compound_seq_len
    val_proteins = data['val_folds_proteins'][fold] # n_fold, n, protein_seq_len
    val_y = data['val_folds_affinity'][fold] # n_fold, n 


    test_drugs = data['test_folds_drugs'][fold] # n_fold, n, compound_seq_len
    test_proteins = data['test_folds_proteins'][fold] # n_fold, n, protein_seq_len
    test_y = data['test_folds_affinity'][fold] # n_fold, n 

    return (train_drugs, train_proteins, train_y), (val_drugs, val_proteins, val_y), (test_drugs, test_proteins, test_y)

def getloaders(dataset, fold, batch_size, mode='train'):
    train_data = getdata(
        dataset_name=dataset,
        fold=fold, mode='train'
    )
    val_data = getdata(
        dataset_name=dataset,
        fold=fold, mode='val'
    )
    test_data = getdata(
        dataset_name=dataset,
        fold=fold, mode='test'
    )
    print("Length of train, val, test: ", len(train_data), len(val_data), len(test_data))
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size = batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size = batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader



def getdata(dataset_name:str, fold:int, mode:str='train'):
    train, val, test = load_mat_file(dataset_name=dataset_name, fold=fold)

    if mode=='train':
        dataset = ProteinCompoudDataset(
            drugs=train[0],
            proteins=train[1],
            y = train[2]
        )
    elif mode=='val':
        dataset = ProteinCompoudDataset(
            drugs=val[0],
            proteins=val[1],
            y = val[2]
        )
    elif mode=='test':
        dataset = ProteinCompoudDataset(
            drugs=test[0],
            proteins=test[1],
            y = test[2]
        )

    else:
        raise ValueError('mode can be train, test or val ')
    return dataset


class ProteinCompoudDataset(Dataset):
    def __init__(
            self,
            drugs: np.ndarray,
            proteins: np.ndarray,
            y: np.ndarray
    ):
        """Dataset for loading compound protein pairs and their affinity

        :param drugs: numpy array of drugs (N, drug_seq_len)
        :param proteins: numpy array of proteins (N, protein_seq_len)
        :param y: numpy array of affinity values (N,)

        :return: self.compound_list[i], self.protein_list[i], self.y[i]
        """
        super().__init__()

        self.compound_list = torch.tensor(drugs, dtype=torch.long)
        self.protein_list = torch.tensor(proteins, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.protein_list)
    
    def __getitem__(self, index):
        return self.compound_list[index,:], self.protein_list[index,:], self.y[index]
     

if __name__=='__main__':
    train_dataset = getdata(
        dataset_name='davis',
        mode = 'train',
        fold=0
    )

    

    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    for c,p,y in loader:
        print(c.shape, p.shape, y.shape)

    
