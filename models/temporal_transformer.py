import math
import torch
from torch import nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=14, d_model=64, heads=4, layers=2, dropout=.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        position = torch.arange(720).unsqueeze(1)
        divisor = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000)/d_model))
        pe = torch.zeros(720,d_model)
        pe[:,0::2], pe[:,1::2] = torch.sin(position*divisor), torch.cos(position*divisor)
        self.register_buffer('position', pe)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, heads, d_model*4, dropout, batch_first=True), layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model,1)

    def forward(self,x,mask):
        h = self.embedding(x) + self.position[:x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = (h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.head(h).squeeze(-1)
