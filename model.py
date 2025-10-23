import torch.nn as nn
import config as default_config

# ######################################################################
# ###########  Building Blocks for the Style-based Generator ###########
# ######################################################################

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) block.
    This layer adjusts the style of the content features based on a given style code.
    """
    def __init__(self, content_channels, style_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(content_channels, affine=False)
        # A linear layer to produce the modulation parameters (gamma and beta)
        self.style_modulation = nn.Linear(style_dim, content_channels * 2)

    def forward(self, content_features, style_code):
        normalized_content = self.instance_norm(content_features)
        
        # Squeeze style_code from [B, D, 1, 1] to [B, D]
        style_params = self.style_modulation(style_code.squeeze(-1).squeeze(-1))
        
        # Split the style parameters into gamma and beta
        gamma, beta = style_params.chunk(2, dim=1)
        
        # Reshape gamma and beta to match content feature dimensions for broadcasting
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        
        # Apply the style modulation
        return gamma * normalized_content + beta

class AdaINResBlk(nn.Module):
    """
    A residual block that incorporates AdaIN layers.
    This allows the style to be injected at multiple points in the generator.
    """
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.adain1 = AdaIN(channels, style_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.adain2 = AdaIN(channels, style_dim)

    def forward(self, x, style_code):
        residual = x
        out = self.relu1(self.adain1(self.conv1(x), style_code))
        out = self.adain2(self.conv2(out), style_code)
        return out + residual

# ######################################################################
# ###################  Main Network Architectures ######################
# ######################################################################

class StyleEncoder(nn.Module):
    """
    Encodes an image into a low-dimensional style vector.
    """
    def __init__(self, style_dim=256):
        super().__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, style_dim, kernel_size=1)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img)


class Generator(nn.Module):
    """
    The main generator network for StyleCycleGAN.
    It separates content and style, encoding the content and then injecting
    the style via AdaIN residual blocks in the decoder.
    """
    def __init__(self, in_channels=3, out_channels=3, style_dim=256, n_residual_blocks=default_config.N_RESIDUAL_BLOCKS):
        super().__init__()
        # Content Encoder: Extracts style-invariant content features
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3, padding_mode='reflect'), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True)
        )
        
        # Decoder: Synthesizes an image from content features and a style code
        decoder_blocks = [AdaINResBlk(256, style_dim) for _ in range(n_residual_blocks)]
        decoder_blocks.extend([
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, 1, 3, padding_mode='reflect'), nn.Tanh()
        ])
        self.decoder = nn.ModuleList(decoder_blocks)

    def forward(self, content_image, style_code):
        content_features = self.content_encoder(content_image)
        x = content_features
        for layer in self.decoder:
            # Apply style code only to the AdaIN residual blocks
            x = layer(x, style_code) if isinstance(layer, AdaINResBlk) else layer(x)
        return x


class Discriminator(nn.Module):
    """
    A PatchGAN-style discriminator.
    It classifies whether image patches are real or fake, providing a more
    stable training signal than a single-value output.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize: 
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False), 
            *block(64, 128), 
            *block(128, 256), 
            *block(256, 512),
            # Final convolution layer to produce a 1-channel output patch
            nn.ZeroPad2d((1, 0, 1, 0)), 
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img): 
        return self.model(img)