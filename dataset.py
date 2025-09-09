import os
import random
import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class StyleTransferDataset(Dataset):
    """
    Dataset for loading source and target domain images for StyleCycleGAN.
    """
    def __init__(self, source_root, target_root, image_size):
        super().__init__()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomChoice([
                transforms.RandomRotation([angle, angle]) for angle in [0, 90, 180, 270]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
            ])

        self.source_files = sorted(glob.glob(os.path.join(source_root, "*.*")))
        self.target_files = sorted(glob.glob(os.path.join(target_root, "*.*")))
        
    def __getitem__(self, index):
        source_img = Image.open(self.source_files[index % len(self.source_files)]).convert('RGB')
        target_img = Image.open(self.target_files[random.randint(0, len(self.target_files) - 1)]).convert('RGB')
        return self.transform(source_img), self.transform(target_img)

    def __len__(self):
        # The number of iterations is determined by the larger of the two domains
        return max(len(self.source_files), len(self.target_files))