from diffuser.models import MLPnet
import torch.nn as nn

class SRModel(nn.Module):
    def __init__(self, in_dim, out_dim, pref_dim, dim=64, dim_mults=(1, 1)) -> None:
        super().__init__()
        self.mlp = MLPnet(in_dim + pref_dim, out_dim, 0, dim, dim_mults, out_act='tanh')
        
    def forward(self, x):
        ''' S, pref, A -> S', R '''
        x = self.mlp(x)
        
        return x