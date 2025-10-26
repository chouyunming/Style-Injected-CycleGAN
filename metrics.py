import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import os
from tqdm import tqdm
import datetime
import config
import traceback

class ImageFolder(Dataset):
    def __init__(self, root_dir, for_metrics=True):
        self.root_dir = root_dir
        self.image_files = []
        
        # Search for images recursively (support both JPG and PNG)
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for root, _, files in os.walk(root_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    self.image_files.append(os.path.join(root, filename))
        
        if not self.image_files:
            raise ValueError(f"No valid images (JPG/PNG) found in {root_dir}")
            
        print(f"Loading {len(self.image_files)} images from {root_dir}")
        
        if for_metrics:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy tensor and path to avoid crashing the whole batch
            return torch.zeros((3, 299, 299), dtype=torch.uint8)

def calculate_metrics(folder_dir, target_dir, device):
    """Calculate FID and KID metrics between a folder and target images"""
    try:
        folder_dataset = ImageFolder(folder_dir, for_metrics=True)
        target_dataset = ImageFolder(target_dir, for_metrics=True)
        
        folder_loader = DataLoader(folder_dataset, batch_size=32, 
                                 num_workers=4, shuffle=False, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=32, 
                                 num_workers=4, shuffle=False, pin_memory=True)
        
        # Initialize metrics
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        
        min_samples = min(len(folder_dataset), len(target_dataset))
        subset_size = max(min(50, min_samples // 2), 2)
        print(f"Using KID subset_size of {subset_size} (total samples: {len(folder_dataset)}/{len(target_dataset)})")
        kid_metric = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
        
        print("Processing generated images...")
        for batch in tqdm(folder_loader, desc="Generated images"):
            images = batch.to(device)
            fid_metric.update(images, real=False)
            kid_metric.update(images, real=False)

        print("Processing real/target images...")
        for batch in tqdm(target_loader, desc="Target images"):
            images = batch.to(device)
            fid_metric.update(images, real=True)
            kid_metric.update(images, real=True)
            
        print("Calculating final scores...")
        fid_score = float(fid_metric.compute())
        kid_mean, _ = kid_metric.compute()
        
        return {
            'FID': fid_score,
            'KID': float(kid_mean) * 1000,
            'folder_count': len(folder_dataset),
            'target_count': len(target_dataset)
        }
        
    except Exception as e:
        print(f"Error during calculation for {folder_dir}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    base_dir = config.METRICS_INPUT_DIR
    target_dir = config.METRICS_TARGET_DIR

    output_filename = f"metrics_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(base_dir, output_filename)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    
    # Calculate metrics for each subdirectory
    results = {}
    print("\nCalculating metrics for all subdirectories...")
    
    for subdir in subdirs:
        print(f"\nProcessing {subdir}...")
        folder_dir = os.path.join(base_dir, subdir)
        metrics = calculate_metrics(folder_dir, target_dir, device)
        if metrics:
            results[subdir] = metrics
    
    # Print and save results
    output_text = []
    # Add timestamp and directory information
    output_text.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_text.append(f"Base directory: {os.path.abspath(base_dir)}")
    output_text.append(f"Target directory: {os.path.abspath(target_dir)}")
    output_text.append("\nResults:")
    output_text.append("Folder        FID ↓      KID (x10³) ↓    Images")
    output_text.append("-" * 55)
    
    best_fid = float('inf')
    best_kid = float('inf')
    best_folder_fid = None
    best_folder_kid = None
    
    # Iterate through results without sorting the folder names
    for folder in results.keys():
        metrics = results[folder]
        result_line = f"{folder:<12} {metrics['FID']:8.2f} {metrics['KID']:14.2f}    {metrics['folder_count']}/{metrics['target_count']}"
        print(result_line)
        output_text.append(result_line)
        
        # Track best performing folder
        if metrics['FID'] < best_fid:
            best_fid = metrics['FID']
            best_folder_fid = folder
        if metrics['KID'] < best_kid:
            best_kid = metrics['KID']
            best_folder_kid = folder
    
    if best_folder_fid is not None:
        output_text.append("\nBest performing folders:")
        output_text.append(f"Best FID: {best_folder_fid} (FID: {best_fid:.2f})")
        output_text.append(f"Best KID: {best_folder_kid} (KID: {best_kid:.2f})")
    
    output_dir_path = os.path.dirname(output_path)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_text))
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()