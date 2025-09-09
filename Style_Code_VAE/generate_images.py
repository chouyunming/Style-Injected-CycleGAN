import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
from tqdm import tqdm
import glob
import random

from trainer import StyleCycleGAN
from train_style_code_vae import StyleCodeVAE

def generate_images(args):
    """
    Generates images by feeding synthetic style codes from a StyleCodeVAE
    into a pre-trained StyleCycleGAN generator.
    """
    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    # --- Load StyleCycleGAN Generator ---
    print(f"Loading StyleCycleGAN model from: {args.gan_checkpoint_dir}")
    gan_model = StyleCycleGAN(device=device, total_epochs=1, lr_g=0, lr_d=0, loss_weights={})
    gan_model.load_models(args.gan_checkpoint_dir)
    
    generator = gan_model.ema_G_A2B if args.use_ema else gan_model.G_A2B
    if args.direction == 'BtoA':
        generator = gan_model.ema_G_B2A if args.use_ema else gan_model.G_B2A
    generator.eval()
    print("StyleCycleGAN Generator loaded successfully.")

    # --- Load StyleCodeVAE Model ---
    print(f"Loading StyleCodeVAE model from: {args.vae_checkpoint_path}")
    vae_model = StyleCodeVAE(
        style_dim=args.vae_style_dim,
        latent_dim=args.vae_latent_dim
    ).to(device)
    try:
        vae_model.load_state_dict(torch.load(args.vae_checkpoint_path, map_location=device))
        vae_model.eval()
    except Exception as e:
        print(f"Error loading StyleCodeVAE state_dict: {e}")
        return
    print("StyleCodeVAE model loaded successfully.")

    # --- Prepare Content Images ---
    os.makedirs(args.output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    content_files = glob.glob(os.path.join(args.content_dir, '*.*'))
    if not content_files:
        print(f"Error: No content images found in {args.content_dir}")
        return

    print(f"Found {len(content_files)} content images. Generating {args.num_images} styled images...")

    # --- Generation Loop ---
    with torch.no_grad():
        for i in tqdm(range(args.num_images), desc="Generating Images"):
            # Select a content image randomly for each generation
            content_path = random.choice(content_files)
            content_image = Image.open(content_path).convert('RGB')
            content_tensor = transform(content_image).unsqueeze(0).to(device)

            # Step 1: Sample a random vector z from the VAE's latent space
            z = torch.randn(1, args.vae_latent_dim).to(device)
            # Step 2: Use the VAE's decoder to generate a new, synthetic style code
            # The output shape is [1, style_dim]
            new_style_code = vae_model.decode(z)
            # Step 3: Reshape the code to [1, style_dim, 1, 1] for the GAN's generator
            style_code_for_gan = new_style_code.unsqueeze(-1).unsqueeze(-1)

            # Generate the final image
            generated_tensor = generator(content_tensor, style_code_for_gan)
            
            # Save the image
            save_image(generated_tensor, os.path.join(args.output_dir, f"gen_{i+1:04d}.png"), normalize=True, value_range=(-1, 1))

    print(f"\nSuccessfully generated and saved {args.num_images} images to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images using a trained StyleCodeVAE and StyleCycleGAN.')
    # --- Paths ---
    parser.add_argument('--gan_checkpoint_dir', type=str, required=True, help='Path to the directory containing the trained StyleCycleGAN model weights.')
    parser.add_argument('--vae_checkpoint_path', type=str, required=True, help='Full path to the StyleCodeVAE model checkpoint (.pth file).')
    parser.add_argument('--content_dir', type=str, required=True, help='Directory containing content images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated images.')
    
    # --- Model Configs ---
    parser.add_argument('--vae_latent_dim', type=int, required=True, help='Latent dimension of the StyleCodeVAE model.')
    parser.add_argument('--vae_style_dim', type=int, required=True, help='Style dimension of the StyleCodeVAE model (must match GAN Style Encoder).')
    parser.add_argument('--direction', type=str, choices=['AtoB', 'BtoA'], default='AtoB', help="Direction of image translation for the GAN.")
    parser.add_argument('--image_size', type=int, default=256, help='Image size used during model training.')
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use the EMA models for inference.')
    
    # --- Generation Configs ---
    parser.add_argument('--num_images', type=int, default=200, help='Number of images to generate.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU).')

    args = parser.parse_args()
    generate_images(args)
