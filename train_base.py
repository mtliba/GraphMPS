import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from autoencoder_training import SimpleAutoEncoder, VariationalAutoEncoder, VQVAE
from gaussian_bias_sampling import GaussianPatchDataset, GaussianCenterBias

# Training function for autoencoders
def train_autoencoder(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Training Variational AutoEncoder (VAE) - Modified training loop due to KL divergence
def train_vae(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            mse_loss = criterion(recon, batch)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + kl_divergence / batch.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Training Vector Quantized VAE (VQ-VAE)
def train_vqvae(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon, z_q = model(batch)
            recon_loss = criterion(recon, batch)
            commitment_loss = torch.mean((z_q.detach() - model.encoder(batch)) ** 2)
            vq_loss = recon_loss + commitment_loss
            vq_loss.backward()
            optimizer.step()
            total_loss += vq_loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Example Usage
height, width = 200, 60
sigma_x, sigma_y = 0.25, 0.7
base_fixations = 200
weight_factor = 2
image_paths = ['/content/img1.png', '/content/img2.png']

# Create dataset and dataloader
gaussian_center_bias = GaussianCenterBias(height, width, sigma_x, sigma_y, base_fixations, weight_factor)
dataset = GaussianPatchDataset(image_paths, gaussian_center_bias)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train Simple AutoEncoder
print("Training Simple AutoEncoder...")
simple_autoencoder = SimpleAutoEncoder()
train_autoencoder(simple_autoencoder, dataloader)

# Train Variational AutoEncoder
print("Training Variational AutoEncoder...")
vae = VariationalAutoEncoder()
train_vae(vae, dataloader)

# Train VQ-VAE
print("Training VQ-VAE...")
vqvae = VQVAE()
train_vqvae(vqvae, dataloader)
