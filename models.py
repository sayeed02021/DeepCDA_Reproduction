import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention_author(nn.Module):
    """PyTorch reimplementation of the author's attention mechanism.
    
    Applied twice:
        - Once on LSTM outputs  (protein_lstm, drug_lstm)
        - Once on CNN outputs   (protein_cnn,  drug_cnn)
    Then GlobalMaxPool each → concat → Add both branches.
    """

    def forward(
        self,
        protein_lstm: torch.Tensor,  # (B, lp, d)
        drug_lstm: torch.Tensor,     # (B, ld, d)
        protein_cnn: torch.Tensor,   # (B, lp', d)  (before LSTM, raw CNN output)
        drug_cnn: torch.Tensor,      # (B, ld', d)
    ) -> torch.Tensor:               # (B, 2d)

        enc1 = self._attend_and_pool(protein_lstm, drug_lstm)   # (B, 2d)
        enc2 = self._attend_and_pool(protein_cnn,  drug_cnn)    # (B, 2d)
        return enc1 + enc2                                       # (B, 2d)  ← Add (skip connection)

    def _attend_and_pool(
        self,
        protein: torch.Tensor,   # (B, lp, d)
        drug: torch.Tensor,      # (B, ld, d)
    ) -> torch.Tensor:           # (B, 2d)

        # --- att_func ---
        # scalar: sigmoid(mean_prot · mean_drug)
        mean_prot = protein.mean(dim=1, keepdim=True)   # (B, 1, d)
        mean_drug = drug.mean(dim=1, keepdim=True)      # (B, 1, d)
        mean_all = torch.sigmoid(
            torch.bmm(mean_prot, mean_drug.transpose(1, 2))  # (B, 1, 1)
        )

        # full attention map: sigmoid(prot @ drug.T) * scalar
        att = torch.sigmoid(
            torch.bmm(protein, drug.transpose(1, 2))    # (B, lp, ld)
        ) * mean_all                                    # broadcast (B, lp, ld)

        # --- coeff_fun_prot ---
        # softmax over ld dim → weighted sum of protein fragments
        att_prot = torch.softmax(
            att.mean(dim=2, keepdim=True),              # (B, lp, 1)
            dim=1
        )                                               # (B, lp, 1)
        weighted_prot = protein * att_prot              # (B, lp, d)

        # --- coeff_fun_lig ---
        # softmax over lp dim → weighted sum of drug fragments
        att_drug = torch.softmax(
            att.mean(dim=1, keepdim=True).transpose(1, 2),  # (B, ld, 1)
            dim=1
        )                                               # (B, ld, 1)
        weighted_drug = drug * att_drug                 # (B, ld, d)

        # --- GlobalMaxPooling ---
        pooled_prot, _ = weighted_prot.max(dim=1)      # (B, d)
        pooled_drug, _ = weighted_drug.max(dim=1)      # (B, d)

        return torch.cat([pooled_prot, pooled_drug], dim=-1)  # (B, 2d)




class CrossAttention_paper(nn.Module):
    """Implementing cross attention given in the paper
    """

    def __init__(
            self,
            lstm_out_dim:int
    ):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(lstm_out_dim,lstm_out_dim)/(lstm_out_dim**0.5))
        self.W2 = nn.Parameter(torch.randn(lstm_out_dim, lstm_out_dim)/(lstm_out_dim**0.5))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.lstm_out_dim = lstm_out_dim

    

    def forward(
            self,
            protein_lstm:torch.Tensor,
            drug_lstm:torch.Tensor
    ):
        """
        :param protein_lstm: (B, lp, d) output of lstm for protein
        :param drug_lstm: (B, ld, d) output of lstm for drug
        
        :return F: (B,2d) fragment descriptor 
        """        
        B,lp,d = protein_lstm.shape
        B,ld,d = drug_lstm.shape

        if d!=self.lstm_out_dim:
            raise ValueError('Output dim of tensors do not match model output dim')

        op_hat = torch.mean(protein_lstm, dim=1, keepdim=True)  # B,1,d  
        od_hat = torch.mean(drug_lstm, dim=1, keepdim=True)      # B,1,d

        alpha1 = torch.einsum('bpd, de, bld->bpl', protein_lstm, self.W1, drug_lstm) # B,lp,ld
        alpha2 = torch.einsum('bid, de, bjd->bij', op_hat, self.W2, od_hat) # B,1,1

        alpha = torch.sigmoid(alpha1)*torch.sigmoid(alpha2) # B,lp,ld


        F_p = torch.einsum('bpl,bpd->bd', alpha, protein_lstm) # B,d
        F_d = torch.einsum('bpl,bld->bd', alpha, drug_lstm) # B,d
        F = torch.cat([F_p, F_d], dim=-1) # B,2d

        return F
    

class FCN(nn.Module):
    """Final connected network for computing affinity
    
    :param in_features: input features
    """
    def __init__(self, in_features):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self, descriptor):
        """
        :param descriptor: (B,D) fragment descriptor
        
        :return affinity: (B,1) predicts the affinity
        """
        affinity = self.fcn(descriptor)
        return affinity
    


class DeepCDA(nn.Module):
    """Drug Target Interaction model. 
    
    :param smiles_dict_len: length of smiles token dictionary
    :param protein_dict_len: length of protein token dictionary
    :param embedding_size: size of embedding dimension
    :param out_dim: channel size after first convolution
    :param protein_k: kernel size for convolution with protein sequences
    :param smiles_k: kernel size for convolution with smiles sequences
    """
    def __init__(
            self,
            smiles_dict_len: int,
            protein_dict_len:  int,
            embedding_size: int, 
            out_dim: int,
            protein_k: int,
            smiles_k: int,
            method_type: str
    ):
        super().__init__()
        
        ## First convolution + LSTM for drugs
        self.method_type = method_type
        self.drug_embedder = nn.Embedding(
            num_embeddings=smiles_dict_len+1,
            embedding_dim=embedding_size,
            padding_idx=0
        )
        self.drug_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=out_dim, kernel_size=smiles_k),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_dim, out_channels=2*out_dim, kernel_size=smiles_k),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*out_dim, out_channels=3*out_dim, kernel_size=smiles_k),
            nn.ReLU()
        )
        self.drug_lstm = nn.LSTM(
            input_size=3*out_dim,
            hidden_size=3*out_dim,
            batch_first=True
        )

        ## Now models for protein
        self.protein_embedder = nn.Embedding(
            num_embeddings=protein_dict_len,
            embedding_dim=embedding_size,
            padding_idx=0
        )
        self.protein_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=out_dim, kernel_size=protein_k),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_dim, out_channels=2*out_dim, kernel_size=protein_k),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*out_dim, out_channels=3*out_dim, kernel_size=protein_k),
            nn.ReLU()
        )
        self.protein_lstm = nn.LSTM(
            input_size=3*out_dim,
            hidden_size=3*out_dim,
            batch_first=True
        )
        if method_type == 'paper':
            self.att = CrossAttention_paper(lstm_out_dim=3*out_dim)
        elif method_type=='github':
            self.att = CrossAttention_author()
        self.pred_layer = FCN(in_features=6*out_dim)

    
    def forward(self, drug_seq, protein_seq) -> torch.Tensor:
        """

        :param drug_seq: token2idx sequences of drugs(B, max_drug_seq_len(L_d)) 
        :param protein_seq: token2idx sequences of proteins(B, max_protein_seq_len(L_p)) 

        :return aff: model's affinity prediction
        """

        drug_embedding = self.drug_embedder(drug_seq) # B,L_d,E
        protein_embedding = self.protein_embedder(protein_seq) # B,L_p,E

        drug_x = self.drug_conv(drug_embedding.permute(0,2,1)) # B, out_dim*3, L_d'
        drug_conv = drug_x.permute(0,2,1) # B, L_d', out_dim*3
        drug_lstm, _ = self.drug_lstm(drug_conv) # B, L_d', out_dim*3


        protein_x = self.protein_conv(protein_embedding.permute(0,2,1))# B, out_dim*3, L_p'
        protein_conv =protein_x.permute(0,2,1) # B, L_p', out_dim*3
        protein_lstm,_ = self.protein_lstm(protein_conv) # B, L_p', out_dim*3

        # print(drug_lstm.shape, protein_lstm.shape)
        if self.method_type=='paper':
            F = self.att(
            protein_lstm, drug_lstm
            )
        elif self.method_type=='github':
            F = self.att(
               protein_lstm, 
               drug_lstm,
               protein_conv,
               drug_conv 
            )
        
        affinity = self.pred_layer(F)
        return F, affinity
    

class Discriminator(nn.Module):
    """
    Discriminator for domain adaptation

    :param in_features: (6*out_dim) dimnesion of input features
    """
    def __init__(self, in_features: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,2)
        )


    def forward(self, F:torch.Tensor) -> torch.Tensor:
        """
        :param F: (B,in_features) encoder output
        :return: (B,1) prob if F is from in distribution or not
        """

        return self.net(F)




        


if __name__=='__main__':

    from DeepCDA_own.dataset import getdata
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch.nn.functional as F
    train_data = getdata(dataset_name='davis', mode='train', fold=0)
    train_loader = DataLoader(train_data, batch_size=128)

    model = DeepCDA(
        smiles_dict_len=64,
        protein_dict_len=25,
        embedding_size=256,
        out_dim=64,
        protein_k=8,
        smiles_k=8
    )

    device = torch.device('mps')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.eval()

    # d,p,a = train_data[0]
    # d,p,a = d.to(device), p.to(device), a.to(device)
    # aff = model(d.unsqueeze(0), p.unsqueeze(0))
    
    for epoch in range(100):
        pbar = tqdm(train_loader, desc=f'Epoch={epoch+1}' ,dynamic_ncols=True, leave=False)
        total_loss = 0
        for idx, (d,p,a) in enumerate(pbar):
            optimizer.zero_grad()
            d,p,a = d.to(device), p.to(device), a.to(device)

            aff = model(
                drug_seq=d,
                protein_seq=p
            )

            # print(a,aff)

            
            loss = F.mse_loss(a, aff.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss+=loss.item()
            pbar.set_postfix(
                {
                    'Loss': total_loss/(idx+1)
                }
            )
        pbar.close()
