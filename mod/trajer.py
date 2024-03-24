import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from diffuser.models import EncoderUnet, MLPnet, DecoderUnet
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

class TrajerIWAE(nn.Module):
    ''' Modified based on https://github.com/AntixK/PyTorch-VAE '''
    def __init__(self, horizon, transition_dim, pref_dim, dim=64, dim_mults=(1, 2, 4), latent_dim=128, kl_weight=.00025, n_samples=5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.traj_shape = (horizon, transition_dim)
        self.pref_dim = pref_dim
        self.alpha = .1
        self.mixup_num = 8
        
        self.enc = EncoderUnet(horizon, transition_dim, pref_dim, dim, dim_mults)
        
        self.hidden_shape = (horizon // (2 ** (len(dim_mults) - 1)), dim_mults[-1] * dim)
        hidden_dim = np.prod(self.hidden_shape)
        print('h_space shape =', self.hidden_shape, 'l_space dim =', latent_dim)
        
        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.var_fc = nn.Linear(hidden_dim, latent_dim)
        self.latent_fc = nn.Linear(latent_dim, hidden_dim)
        
        # self.mu_fc = nn.Linear(hidden_dim, hidden_dim)
        # self.var_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.dec = DecoderUnet(horizon, transition_dim, pref_dim, dim, dim_mults)
        
    def _reparam(self, mu, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + std * eps
    
    def _mixup(self, x, pref):
        ''' x: [BxS x H x T], pref: [BxS x n_obj] '''
        lamda = np.random.beta(self.alpha, self.alpha)
        idx = np.random.randint(0, len(x), 2 * self.mixup_num)
        idx1, idx2 = idx[:self.mixup_num], idx[self.mixup_num:]
        x_tilde = lamda * x[idx1] + (1 - lamda) * x[idx2]
        pref_tilde = lamda * pref[idx1] + (1-lamda) * pref[idx2]
        
        x = torch.cat([x, x_tilde], dim=0)
        pref = torch.cat([pref, pref_tilde], dim=0)
        
        return x, pref
    
    def forward(self, x, pref): # Reconstuction
        ''' x: [B x S x H x T], pref: [B x S x n_obj] '''
        b, s = x.shape[0], x.shape[1]
        x = x.reshape(-1, *self.traj_shape)
        pref = pref.reshape(-1, self.pref_dim)
        
        hidden = self.enc(x, pref).flatten(-2)
        mu, log_var = self.mu_fc(hidden), self.var_fc(hidden)
        
        latent = self._reparam(mu, log_var) # bottleneck layer latent
        z = self.latent_fc(latent).reshape(-1, *self.hidden_shape)
        # z = self._reparam(mu, log_var).reshape(-1, *self.hidden_shape) # bottleneck layer latent
        
        recon = self.dec(z, pref).view(b, s, *self.traj_shape)
        mu = mu.view(b, s, -1)
        log_var = log_var.view(b, s, -1)
        
        return recon, mu, log_var

    @torch.no_grad()
    def sample(self, pref):
        z = torch.randn((len(pref), self.latent_dim), dtype=torch.float32, device=pref.device)
        z = self.latent_fc(z).reshape(-1, *self.hidden_shape)
        recon = self.dec(z, pref)
        
        return recon
    
    @torch.no_grad()
    def generate(self, x, pref): # Reconstruction w/o gradients
        x = x.unsqueeze(1)
        return self.forward(x, pref)[0]

    def loss(self, x, pref):
        x = x.repeat(self.n_samples, 1, 1, 1).transpose(0, 1) # [B x S x H x T]
        pref = pref.repeat(self.n_samples, 1, 1).transpose(0, 1) # [B x S x n_obj]
        x, pref = self._mixup(x, pref)
        x_recon, mu, log_var = self.forward(x, pref)
        
        log_p_x_z = ((x_recon - x) ** 2).flatten(-2).mean(-1) # reconstruction loss [B x S]
        loss_kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2) # kl loss [B x S]
        
        log_weight = log_p_x_z + loss_kl * self.kl_weight # Importance Weight [B x S]
        iw = F.softmax(log_weight, dim=-1) # [B]
        
        loss = torch.mean(torch.sum(iw * log_weight, dim=-1), dim=0)
        
        info = {
            'loss': loss.item(),
            'recon': log_p_x_z.mean().item(),
            'kl': -loss_kl.mean().item(),
        }
        return loss, info