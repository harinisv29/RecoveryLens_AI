import torch
from torch import nn

class GRUClassifier(nn.Module):
    def __init__(self,input_dim=14):
        super().__init__()
        self.gru=nn.GRU(input_dim,64,batch_first=True)
        self.head=nn.Linear(64,1)
    def forward(self,x,mask):
        packed=nn.utils.rnn.pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed)
        return self.head(h[-1]).squeeze(-1)
