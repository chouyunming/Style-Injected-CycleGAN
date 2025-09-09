import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image


def add_text_to_image(img_tensor, text):
    """
    Adds text with a shadow to a PyTorch image tensor for better visibility.
    """
    img_np = img_tensor.cpu().float().numpy()
    # Convert from the range [-1, 1] to [0, 1]
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # A larger default font
        font = ImageFont.load_default(size=15)
    except (AttributeError, TypeError): 
        # Fallback for older Pillow versions
        font = ImageFont.load_default()
    
    text_color = "white"
    shadow_color = "black"
    x, y = 10, 10
    # Add a simple shadow by drawing the text in black at offset positions
    for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + offset[0], y + offset[1]), text, font=font, fill=shadow_color)
    # Draw the main text
    draw.text((x, y), text, font=font, fill=text_color)
    
    img_np = np.array(img_pil) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.from_numpy(img_np).float()
    # Convert back to the range [-1, 1] for consistency
    return (img_tensor * 2 - 1)


def save_sample_grid(real_A, fake_B, real_B, fake_A, path):
    """
    Saves a 2x2 grid of labeled image samples (Real A, Fake B, Real B, Fake A).
    """
    real_A_labeled = add_text_to_image(real_A[0], "Real A")
    fake_B_labeled = add_text_to_image(fake_B[0], "Fake B")
    real_B_labeled = add_text_to_image(real_B[0], "Real B")
    fake_A_labeled = add_text_to_image(fake_A[0], "Fake A")

    # Stack the four images to create a grid
    samples = torch.stack([real_A_labeled, fake_B_labeled, real_B_labeled, fake_A_labeled])
    save_image(samples, path, nrow=2, normalize=True, value_range=(-1, 1))

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Applying EMA to the generator weights can lead to more stable and higher-quality
    image generation during inference.
    """
    def __init__(self, beta):
        self.beta = beta
    
    def update_model_average(self, ma_model, current_model):
        """
        Update the moving average model with the current model's parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DynamicWeightScheduler:
    """
    Dynamically adjusts the weights of different loss components during training.
    This scheduler implements a warmup phase followed by a cosine decay phase,
    which can help stabilize training in the early stages and refine the model later.
    """
    def __init__(self, init_weights, warmup_epochs=10, decay_epochs=100, total_epochs=200):
        self.init_weights = init_weights
        self.current_weights = init_weights.copy()
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = warmup_epochs + decay_epochs
        self.total_epochs = total_epochs
        
        self.loss_history = {k: [] for k in init_weights.keys()}
        self.weight_history = {k: [] for k in init_weights.keys()}

    def get_current_weights(self, epoch, current_losses):
        # 1. Update loss history
        for k, v in current_losses.items():
            if k in self.loss_history:
                self.loss_history[k].append(v.detach().cpu().item())

        # 2. Calculate warmup factor
        warmup_factor = min(1.0, (epoch + 1) / self.warmup_epochs)

        # 3. Calculate decay factor (Cosine decay)
        decay_factor = 1.0
        if epoch >= self.warmup_epochs:
            progress = min(1.0, (epoch - self.warmup_epochs) / (self.decay_end_epoch - self.warmup_epochs))
            # Cosine decay from 1 down to 0
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            # Rescale to decay from 1 down to a minimum of 0.1
            decay_factor = 0.1 + 0.9 * cosine_decay

        # 4. Update weights and record history
        for k in self.current_weights.keys():
            self.current_weights[k] = self.init_weights[k] * warmup_factor * decay_factor
            self.weight_history[k].append(self.current_weights[k])
            
        return self.current_weights

    def plot_weight_history(self, save_path=None):
        """
        Plots the evolution of the loss weights over epochs and saves the plot.
        """
        if not any(self.weight_history.values()):
            return
        plt.figure(figsize=(15, 8))
        for k, v in self.weight_history.items():
            if v:
                plt.plot(v, label=k, linewidth=2)
        plt.title('Loss Weight Evolution Over Training')
        plt.xlabel('Epochs')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()