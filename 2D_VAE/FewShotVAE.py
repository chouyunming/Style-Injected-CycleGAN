import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse  # Import the argparse library

import config

class VAEDataset(Dataset):
    """
    Custom Dataset for loading images from a single directory for VAE training.
    """
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomChoice([
                transforms.RandomRotation([angle, angle]) for angle in [0, 90, 180, 270]]),            
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
        if not self.image_files:
            raise ValueError(f"No images found in the directory: {root_dir}")
        print(f"Found {len(self.image_files)} images in {root_dir}")
        if len(self.image_files) < 50:
            print(f"Warning: Dataset is very small ({len(self.image_files)} images).")

    def __len__(self):
        if len(self.image_files) < 50:
            return len(self.image_files) * 100
        return len(self.image_files)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_files)
        img_path = self.image_files[actual_idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except (IOError, OSError) as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return torch.zeros((3, config.IMAGE_SIZE, config.IMAGE_SIZE))


class VAE(nn.Module):
    """
    Lightweight VAE designed for few-shot learning.
    """
    def __init__(self, latent_dim, in_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        encoder_output_size = 256 * 16 * 16
        self.fc_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, encoder_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_div) / x.size(0)

# --- MODIFICATION: The main function now accepts 'args' from the parser ---
def main(args):
    """
    Main function to set up and run the VAE training process.
    """
    # --- Get parameters from command-line arguments ---
    latent_dim = args.latent_dim
    beta = args.beta

    # --- Setup ---
    device = torch.device(f"cuda:{config.GPU}" if torch.cuda.is_available() else "cpu")
    # Create a unique experiment folder based on the parameters
    experiment_name = f"vae_fewshot_beta{beta}_ld{latent_dim}"
    output_dir = os.path.join(config.SAVE_DIR_BASE, experiment_name)
    samples_dir = os.path.join(output_dir, "samples")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output will be saved to: {output_dir}")

    # --- Data Loading ---
    try:
        dataset = VAEDataset(root_dir=config.TARGET_DIR, image_size=config.IMAGE_SIZE)
        dataloader = DataLoader(
            dataset, BATCH_SIZE=config.BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return

    # --- Model and Optimizer ---
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Training started... Beta: {beta}, Latent Dim: {latent_dim}")

    # --- Training Loop ---
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for real_images in progress_bar:
            real_images = real_images.to(device)

            recon_images, mu, logvar = model(real_images)
            loss = vae_loss_function(recon_images, real_images, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == config.NUM_EPOCHS - 1:
             print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0 or epoch == config.NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                recon_batch, _, _ = model(real_images)
                comparison = torch.cat([real_images[:8], recon_batch[:8]])
                save_image(comparison.cpu(), os.path.join(samples_dir, f'reconstruction_epoch_{epoch+1}.png'), nrow=8, normalize=True)

                z_sample = torch.randn(config.BATCH_SIZE, model.latent_dim).to(device)
                generated_samples = model.decode(z_sample)
                save_image(generated_samples.cpu(), os.path.join(samples_dir, f'generated_epoch_{epoch+1}.png'), normalize=True)
            
            checkpoint_path = os.path.join(checkpoints_dir, f"vae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint and sample images for epoch {epoch+1}")

    print("Training complete.")

# --- MODIFICATION: Add this block to parse command-line arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE for few-shot learning.')
    parser.add_argument('--beta', type=float, default=1.0, help='The beta value for the KL divergence weight.')
    parser.add_argument('--latent_dim', type=int, default=8, help='The dimension of the latent space.')
    args = parser.parse_args()
    main(args)
