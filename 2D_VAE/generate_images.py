import torch
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import re
from FewShotVAE import VAE
import config

def process_experiment(experiment_dir, output_dir, epoch, num_images, device):
    """
    Processes a single experiment directory to generate images.
    """
    # --- Step 1: Parse parameters from the directory name ---
    experiment_name = os.path.basename(experiment_dir.rstrip('/\\'))
    match = re.search(r"beta([\d\.]+)_ld(\d+)", experiment_name)
    
    if not match:
        print(f"Skipping directory, as it does not match expected format: '{experiment_name}'")
        return
        
    # Extract latent_dim from the folder name
    latent_dim = int(match.group(2))
    print(f"Processing: {experiment_name} | Parsed latent_dim: {latent_dim}")

    # --- Step 2: Construct paths ---
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    # New output directory structure
    output_dir = os.path.join(output_dir, experiment_name)

    # Check if checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Skipping.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"  -> Generating images from: {checkpoint_path}")
    print(f"  -> Saving generated images to: {output_dir}")

    # --- Load Model ---
    model = VAE(latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict for {experiment_name}: {e}")
        return
        
    model.eval()

    # --- Generation Loop ---
    with torch.no_grad():
        for i in tqdm(range(num_images), desc=f"Generating for {experiment_name}", leave=False):
            z = torch.randn(1, latent_dim).to(device)
            generated_sample = model.decode(z)
            save_image(generated_sample, os.path.join(output_dir, f"gen_{i+1:04d}.png"), normalize=True)

    print(f"  -> Successfully generated and saved {num_images} images for {experiment_name}.")


def main(args):
    """
    Main function to find all experiment subdirectories and generate images for each.
    """
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get all subdirectories in the base results directory
    try:
        subdirectories = [f.path for f in os.scandir(args.base_dir) if f.is_dir()]
    except FileNotFoundError:
        print(f"Error: Base results directory not found at '{args.base_dir}'")
        return

    if not subdirectories:
        print(f"No experiment subdirectories found in '{args.base_dir}'.")
        return
        
    print(f"Found {len(subdirectories)} experiment directories to process.")

    # Loop through each experiment subdirectory
    for exp_dir in tqdm(subdirectories, desc="Overall Progress"):
        process_experiment(
            experiment_dir=exp_dir,
            output_dir=args.output_dir,
            epoch=args.epoch,
            num_images=args.num_images,
            device=device
        )
        print("-" * 50)

    print("\nAll experiments processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate images from all VAE experiment folders in a directory.')
    # --- MODIFIED: Updated arguments for batch processing ---
    parser.add_argument('--base_dir', type=str, default=config.SAVE_DIR_BASE, help='Path to the parent directory containing all experiment folders (e.g., ./stylecyclegan/VAE/results).')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR_BASE, help='Path to the parent directory where all generated image folders will be saved (e.g., ./stylecyclegan/VAE/output).')
    parser.add_argument('--epoch', type=int, default=1000, help='The checkpoint epoch number to use for all experiments.')
    parser.add_argument('--num_images', type=int, default=200, help='Number of images to generate for each experiment.')
    args = parser.parse_args()
    main(args)
