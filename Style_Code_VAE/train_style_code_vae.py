import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import glob
from PIL import Image

# Assume these modules are in an importable path
from trainer import StyleCycleGAN
import config as default_config

class StyleCodeVAE(nn.Module):
    """
    A simple MLP-VAE for learning the distribution of style codes.
    """
    def __init__(self, style_dim=64, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_div) / x.size(0)

def main():
    """
    Main function to run the grid search for training StyleCodeVAE.
    Reads all parameters from config.py.
    """
    device = torch.device(f'cuda:{default_config.GPU}' if torch.cuda.is_available() and default_config.GPU >= 0 else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load StyleCycleGAN and extract style codes (done once) ---
    print("Loading StyleCycleGAN to extract style codes...")
    gan_model = StyleCycleGAN(device=device, total_epochs=1, lr_g=0, lr_d=0, loss_weights={})
    gan_model.load_models(default_config.STYLE_ENCODER_CHECKPOINT)
    
    style_encoder = gan_model.ema_SE_B if default_config.INFERENCE_USE_EMA else gan_model.SE_B
    if default_config.INFERENCE_DIRECTION == 'BtoA':
        style_encoder = gan_model.ema_SE_A if default_config.INFERENCE_USE_EMA else gan_model.SE_A
    style_encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((default_config.IMAGE_SIZE, default_config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    style_files = glob.glob(os.path.join(default_config.TARGET_DIR, '*.*'))
    if not style_files:
        print(f"Error: No style images found in {default_config.TARGET_DIR}")
        return

    print(f"Found {len(style_files)} images to create style code dataset.")
    style_code_list = []
    with torch.no_grad():
        for f in style_files:
            img = Image.open(f).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            code = style_encoder(tensor)
            style_code_list.append(code)
    
    style_codes = torch.cat(style_code_list, dim=0)
    
    # --- FIX: Reshape the style codes from [N, C, 1, 1] to [N, C] ---
    if style_codes.dim() == 4:
        style_codes = style_codes.squeeze(-1).squeeze(-1)

    style_dim = style_codes.shape[1]
    print(f"Created style code dataset with shape: {style_codes.shape}")

    dataset = TensorDataset(style_codes)
    dataloader = DataLoader(dataset, batch_size=len(style_codes), shuffle=True)

    # --- Step 2: Run Grid Search for VAE Training ---
    latent_dim_values = default_config.LATENT_DIM
    
    for beta in default_config.BETA_VALUES:
        for latent_dim in latent_dim_values:
            print("\n" + "="*60)
            print(f"STARTING EXPERIMENT: Beta = {beta}, Latent_Dim = {latent_dim}")
            print("="*60)

            vae_model = StyleCodeVAE(style_dim=style_dim, latent_dim=latent_dim).to(device)
            optimizer = optim.Adam(vae_model.parameters(), lr=1e-4)

            for epoch in tqdm(range(default_config.SV_NUM_EPOCHS), desc=f"Training Beta={beta}, LD={latent_dim}"):
                for data in dataloader:
                    codes = data[0].to(device)
                    recon_codes, mu, logvar = vae_model(codes)
                    loss = vae_loss_function(recon_codes, codes, mu, logvar, beta=beta)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            print(f"Final Loss for Beta={beta}, LD={latent_dim}: {loss.item():.4f}")

            # --- Step 3: Save the trained VAE model ---
            output_dir = default_config.SV_SAVE_DIR_BASE
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"stylecodevae_beta{beta}_ld{latent_dim}.pth"
            output_path = os.path.join(output_dir, output_filename)
            
            torch.save(vae_model.state_dict(), output_path)
            print(f"Training complete. VAE model saved to: {output_path}")

    print("\nGrid search finished for all parameter combinations.")


if __name__ == "__main__":
    main()