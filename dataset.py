import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model_base import SimpleAutoEncoder, VariationalAutoEncoder, VQVAE
from dataset import GaussianPatchDataset

# Function to save training loss plot
def save_loss_plot(train_losses, epoch, model_name, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for {model_name} - Epoch {epoch}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_epoch_{epoch}.png"))
    plt.close()

# Function to save reconstructed images plot
def save_reconstructed_images(original_images, reconstructed_images, epoch, model_name, output_dir="./plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_images = min(8, len(original_images))
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].permute(1, 2, 0).numpy())
        plt.title("Original")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_reconstructed_epoch_{epoch}.png"))
    plt.close()

# Training function for autoencoders with added metrics and plotting
def train_autoencoder(model, dataloader, epochs=10, lr=1e-3, model_name="AutoEncoder"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save training loss plot
        save_loss_plot(train_losses, epoch + 1, model_name)

        # Save reconstructed images plot
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                original_images = batch
                reconstructed_images = model(batch)
                break  # Only plot the first batch
        save_reconstructed_images(original_images, reconstructed_images, epoch + 1, model_name)

# Evaluation function to plot some patches of an image
def evaluate_model(model, dataloader, model_name="AutoEncoder", output_dir="./eval_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            original_images = batch
            reconstructed_images = model(batch)
            break  # Only evaluate the first batch

    num_images = min(8, len(original_images))
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].permute(1, 2, 0).numpy())
        plt.title("Original")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_evaluation.png"))
    plt.close()

# Training Variational AutoEncoder (VAE) with additional metrics and plotting
def train_vae(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    train_losses = []
    
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
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save training loss plot
        save_loss_plot(train_losses, epoch + 1, "Variational AutoEncoder")

        # Save reconstructed images plot
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                original_images = batch
                reconstructed_images, _, _ = model(batch)
                break  # Only plot the first batch
        save_reconstructed_images(original_images, reconstructed_images, epoch + 1, "Variational AutoEncoder")

# Training Vector Quantized VAE (VQ-VAE) with additional metrics and plotting
def train_vqvae(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    train_losses = []
    
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
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save training loss plot
        save_loss_plot(train_losses, epoch + 1, "VQ-VAE")

        # Save reconstructed images plot
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                original_images = batch
                reconstructed_images, _ = model(batch)
                break  # Only plot the first batch
        save_reconstructed_images(original_images, reconstructed_images, epoch + 1, "VQ-VAE")

# Example Usage
height, width = 200, 60
sigma_x, sigma_y = 0.25, 0.7
base_fixations = 200
weight_factor = 2
image_paths = ['/content/img1.png', '/content/img2.png']

# Create dataset and dataloader
dataset = GaussianPatchDataset(image_paths, height=height, width=width, sigma_x=sigma_x, sigma_y=sigma_y, base_fixations=base_fixations, weight_factor=weight_factor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train Simple AutoEncoder
print("Training Simple AutoEncoder...")
simple_autoencoder = SimpleAutoEncoder()
train_autoencoder(simple_autoencoder, dataloader, model_name="Simple AutoEncoder")

# Train Variational AutoEncoder
print("Training Variational AutoEncoder...")
vae = VariationalAutoEncoder()
train_vae(vae, dataloader)

# Train VQ-VAE
print("Training VQ-VAE...")
vqvae = VQVAE()
train_vqvae(vqvae, dataloader)

# Evaluate models
print("Evaluating Simple AutoEncoder...")
evaluate_model(simple_autoencoder, dataloader, model_name="Simple AutoEncoder")

print("Evaluating Variational AutoEncoder...")
evaluate_model(vae, dataloader, model_name="Variational AutoEncoder")

print("Evaluating VQ-VAE...")
evaluate_model(vqvae, dataloader, model_name="VQ-VAE")
