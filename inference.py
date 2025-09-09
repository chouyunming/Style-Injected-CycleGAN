# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
from tqdm import tqdm
import glob
import random
import re
from trainer import StyleCycleGAN
import config as default_config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from Style_Code_VAE.train_style_code_vae import StyleCodeVAE
except ImportError:
    print("Warning: Could not import StyleCodeVAE. 'vae' mode will be unavailable.")
    StyleCodeVAE = None


def parse_vae_params_from_filename(checkpoint_path):
    """
    Parse beta and latent_dim from the VAE checkpoint filename.
    Expected format: stylecodevae_beta{X.X}_ld{Y}.pth
    
    Returns:
        tuple: (beta, latent_dim) or (None, None) if parsing fails
    """
    filename = os.path.basename(checkpoint_path)
    
    # Pattern to match: stylecodevae_beta{number}_ld{number}.pth
    pattern = r'stylecodevae_beta([0-9.]+)_ld([0-9]+)\.pth'
    match = re.match(pattern, filename)
    
    if match:
        beta = float(match.group(1))
        latent_dim = int(match.group(2))
        return beta, latent_dim
    else:
        print(f"Warning: Could not parse beta and latent_dim from filename: {filename}")
        return None, None


def load_style_code_vae_model(args, device):
    """
    Loads the trained StyleCodeVAE model with automatic parameter detection.
    """
    if StyleCodeVAE is None:
        print("Error: StyleCodeVAE class was not imported, cannot load the model.")
        return None

    checkpoint_path = args.vae_checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: VAE checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading StyleCodeVAE model weights from: {checkpoint_path}")
    
    # Try to parse parameters from filename first
    beta_from_file, latent_dim_from_file = parse_vae_params_from_filename(checkpoint_path)
    
    # Use parameters from filename if available, otherwise fall back to config/args
    if latent_dim_from_file is not None:
        latent_dim = latent_dim_from_file
        print(f"Detected latent_dim from filename: {latent_dim}")
    else:
        latent_dim = args.vae_latent_dim
        print(f"Using latent_dim from config/args: {latent_dim}")
    
    # Initialize the VAE model with the correct dimensions
    model = StyleCodeVAE(
        style_dim=args.vae_style_dim,
        latent_dim=latent_dim
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        print("StyleCodeVAE model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the StyleCodeVAE model weights: {e}")
        
        # If automatic detection failed, suggest the correct parameters
        if latent_dim_from_file is not None and latent_dim_from_file != args.vae_latent_dim:
            print(f"Suggestion: The checkpoint appears to be trained with latent_dim={latent_dim_from_file}, ")
            print(f"but the current configuration uses latent_dim={args.vae_latent_dim}.")
            print(f"Please update INFERENCE_VAE_LATENT_DIM in config.py to {latent_dim_from_file}")
        
        return None


def preload_style_vectors(args, model, device):
    """
    Preload and return all style codes from a directory of target domain images.
    """
    if not os.path.isdir(args.target_domain_dir):
        print(f"ERROR: '{args.target_domain_dir}' is not a valid target domain path.")
        return None

    use_ema = args.use_ema
    style_encoder = model.ema_SE_B if use_ema else model.SE_B
    if args.direction == 'BtoA':
        style_encoder = model.ema_SE_A if use_ema else model.SE_A
    style_encoder.eval()

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
    style_files = []
    for ext in image_extensions:
        style_files.extend(glob.glob(os.path.join(args.target_domain_dir, ext)))

    if not style_files:
        print(f"Style images not found in '{args.target_domain_dir}'.")
        return None

    print(f"Found {len(style_files)} style images. Preloading all style vectors...")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    all_style_vectors = []
    with torch.no_grad():
        for style_path in tqdm(style_files, desc="Encoding all styles..."):
            style_image = Image.open(style_path).convert('RGB')
            style_tensor = transform(style_image).unsqueeze(0).to(device)
            code = style_encoder(style_tensor)
            all_style_vectors.append(code)

    if not all_style_vectors:
        print("Failure to extract any style encoding from the target domain image.")
        return None

    print(f"Successfully preloaded {len(all_style_vectors)} style vectors.")
    return all_style_vectors


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")

    # --- Load StyleCycleGAN model ---
    print("Building StyleCycleGAN model architecture...")
    model = StyleCycleGAN(device=device, total_epochs=1, lr_g=0, lr_d=0, loss_weights={})
    model.load_models(args.checkpoint_dir)
    print("StyleCycleGAN model weights loaded successfully.")

    # --- Prepare style source based on style_mode ---
    vae_model = None
    all_style_vectors = None
    if args.style_mode == 'vae':
        vae_model = load_style_code_vae_model(args, device)
        if vae_model is None:
            print("Could not load VAE model, terminating program.")
            return
    else:
        all_style_vectors = preload_style_vectors(args, model, device)
        if not all_style_vectors:
            print("Could not load style vectors, terminating program.")
            return
        
    # --- Prepare Generator and Content Images ---
    use_ema = args.use_ema
    generator = model.ema_G_A2B if use_ema else model.G_A2B
    if args.direction == 'BtoA':
        generator = model.ema_G_B2A if use_ema else model.G_B2A
    generator.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    content_files = glob.glob(os.path.join(args.input_dir, '*.*'))
    print(f"Found {len(content_files)} content images. Applying style using '{args.style_mode}' mode.")

    if args.style_mode == 'average':
        fixed_style_code = torch.mean(torch.stack(all_style_vectors), dim=0)

    # --- Run Inference ---
    with torch.no_grad():
        for content_path in tqdm(content_files, desc=f"Applying style via '{args.style_mode}' mode"):
            # --- Dynamic style code generation ---
            if args.style_mode == 'average':
                current_style_code = fixed_style_code
            elif args.style_mode == 'random':
                current_style_code = random.choice(all_style_vectors)
            elif args.style_mode == 'interpolate':
                s_A, s_B = random.sample(all_style_vectors, 2)
                alpha = random.random()
                current_style_code = alpha * s_A + (1.0 - alpha) * s_B
            elif args.style_mode == 'noise':
                s = random.choice(all_style_vectors)
                noise = torch.randn_like(s) * args.noise_level
                current_style_code = s + noise
            elif args.style_mode == 'vae':
                # Get the actual latent dimension from the loaded VAE model
                actual_latent_dim = vae_model.latent_dim
                # Step 1: Sample a random vector z from the VAE's latent space
                z = torch.randn(1, actual_latent_dim).to(device)
                # Step 2: Use the VAE's decoder to generate a new, synthetic style code
                # The output shape is [1, style_dim]
                new_style_code = vae_model.decode(z)
                # Step 3: Reshape the code to [1, style_dim, 1, 1] for the GAN's generator
                current_style_code = new_style_code.unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f"Unknown style mode: {args.style_mode}")

            content_image = Image.open(content_path).convert('RGB')
            content_tensor = transform(content_image).unsqueeze(0).to(device)
            generated_tensor = generator(content_tensor, current_style_code)

            content_base, _ = os.path.splitext(os.path.basename(content_path))
            output_filename = f"{content_base}.png"
            save_image(generated_tensor, os.path.join(args.output_dir, output_filename), normalize=True, value_range=(-1, 1))
    
    print(f"\nInference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StyleCycleGAN Inference with advanced style sampling')
    # --- StyleCycleGAN Parameters ---
    parser.add_argument('--input_dir', type=str, default=default_config.INFERENCE_INPUT_DIR)
    parser.add_argument('--target_domain_dir', type=str, default=default_config.TARGET_DIR)
    parser.add_argument('--output_dir', type=str, default=default_config.INFERENCE_OUTPUT_DIR)
    parser.add_argument('--checkpoint_dir', type=str, default=default_config.INFERENCE_CHECKPOINT_DIR)
    parser.add_argument('--direction', type=str, choices=['AtoB', 'BtoA'], default=default_config.INFERENCE_DIRECTION)
    parser.add_argument('--gpu', type=int, default=default_config.GPU)
    parser.add_argument('--image_size', type=int, default=default_config.IMAGE_SIZE)
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # --- Style Mode Selection ---
    parser.add_argument('--style_mode', type=str, default='average',
                        choices=['average', 'random', 'interpolate', 'noise', 'vae'],
                        help='Style generation mode for inference.')
    parser.add_argument('--noise_level', type=float, default=0.1, help="Noise level for the 'noise' style mode.")

    # --- VAE Parameters (only used if style_mode is 'vae') ---
    parser.add_argument('--vae_checkpoint_path', type=str, default=getattr(default_config, 'INFERENCE_VAE_CHECKPOINT', ''), help='Path to the trained StyleCodeVAE checkpoint.')
    parser.add_argument('--vae_latent_dim', type=int, default=getattr(default_config, 'INFERENCE_VAE_LATENT_DIM', 16), help='Latent dimension of the StyleCodeVAE model.')
    parser.add_argument('--vae_style_dim', type=int, default=getattr(default_config, 'INFERENCE_STYLE_DIM', 256), help='Style dimension of the StyleCodeVAE model.')

    args = parser.parse_args()
    main(args)