import torch
from torch import nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh()):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.linear(x)) + x


class HeartbeatMLP(nn.Module):
    
    def __init__(self, num_harmonics=100, hidden=128, num_beats=500, latent_dim=8):
        super().__init__()
        self.num_harmonics = num_harmonics

        # Spatial variance embedding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

        # Beat embedding
        self.beat_embedding = nn.Embedding(num_beats, latent_dim)
        nn.init.normal_(self.beat_embedding.weight, std=0.01)

        # FFT embedding
        self.input_proj  = nn.Sequential(
            nn.Linear(2 * num_harmonics + latent_dim + 8, hidden),
            nn.Tanh(),
        )
        nn.init.normal_(self.input_proj[0].weight, std=0.01)
        nn.init.zeros_(self.input_proj[0].bias)

        self.res_block_1 = ResidualBlock(hidden, activation=nn.Tanh())
        self.res_block_2 = ResidualBlock(hidden, activation=nn.Tanh())
        self.output_layer = nn.Linear(hidden, 1)

        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, phase, beat_idx, skin_xyz_norm):
        harmonics = torch.arange(1, self.num_harmonics + 1, device=phase.device).float()
        x = phase * harmonics          
        x_periodic = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [1, 2*num_freqs]

        beat_emb = self.beat_embedding(beat_idx)

        M = skin_xyz_norm.shape[0]
        x_global = torch.cat([x_periodic, beat_emb], dim=-1).expand(M, -1)  # [1, 2K+8]
        x_spatial = self.spatial_encoder(skin_xyz_norm) # [M, 8]
        x = torch.cat([x_global, x_spatial], dim=-1)
         
        x = self.input_proj(x)  
        x = self.res_block_1(x)  
        x = self.res_block_2(x)     
        x = self.output_layer(x)    

        return x