import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Simple Convolutional AutoEncoder
class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x ,reconstruct = True):
        encoded = self.encoder(x)
        if reconstruct : 
          decoded = self.decoder(encoded)
          return decoded
        else : 
          return encoded

# Variational AutoEncoder (VAE)
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, 128)
        self.fc_logvar = nn.Linear(256 * 4 * 4, 128)
        self.fc_decode = nn.Linear(128, 256 * 4 * 4)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x,reconstruct = True):
        encoded = self.encoder(x).view(x.size(0), -1)
        if reconstruct : 
          mu = self.fc_mu(encoded)
          logvar = self.fc_logvar(encoded)
          z = self.reparameterize(mu, logvar)
          decoded = self.decoder(self.fc_decode(z).view(-1, 256, 4, 4))
          return decoded, mu, logvar
        else : 
          return encoded
        

# Vector Quantized VAE (VQ-VAE)
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x ,reconstruct = True):
        z_e = self.encoder(x)
        z_e_flattened = z_e.view(-1, z_e.size(1))
        d = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.codebook.weight ** 2, dim=1) - 2 * torch.matmul(z_e_flattened, self.codebook.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_indices).view(z_e.shape)
        if reconstruct : 
          x_recon = self.decoder(z_q)
          return x_recon, z_q
        else : 
          return z_q


