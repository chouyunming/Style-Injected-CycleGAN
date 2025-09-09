import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGStyleContentLoss(nn.Module):
    """
    Computes perceptual style and content losses using a pre-trained VGG19 network.
    The style loss is based on the Gram matrix of feature maps, while the content
    loss is the L1 distance between feature maps.
    """
    def __init__(self, device):
        super().__init__()
        # Using the recommended VGG19 weights
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        
        # Names of the layers to be used for style and content extraction
        self.content_layers_default = ['relu_4_1'] 
        self.style_layers_default = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']

        # Build the VGG model layer by layer and name them
        self.vgg_layers = nn.ModuleDict()
        i, j = 0, 0
        for layer in vgg19.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                j = 1
                name = f'conv_{i}_{j}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}_{j}'
                # Use non-inplace ReLU to allow multiple uses of the same layer
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                j = 0
                name = f'pool_{i}'
            else:
                # Handle unexpected layer types
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            self.vgg_layers[name] = layer
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Pre-calculated ImageNet mean and std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _normalize(self, tensor):
        """
        Normalize a tensor from [-1, 1] to the ImageNet distribution.
        """
        # First, shift from [-1, 1] to [0, 1]
        tensor_01 = (tensor + 1) / 2
        # Then, apply ImageNet normalization
        return (tensor_01 - self.mean) / self.std

    def get_features(self, image, layers):
        """
        Extract features from the specified VGG layers.
        """
        features = {}
        x = image
        for name, layer in self.vgg_layers.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features

    def compute_gram_matrix(self, input_tensor):
        """
        Compute the Gram matrix of a batch of feature maps.
        """
        a, b, c, d = input_tensor.size()  # (batch, channels, height, width)
        features = input_tensor.view(a * b, c * d)
        G = torch.mm(features, features.t())
        # Normalize by the number of elements in the feature map
        return G.div(a * b * c * d)

    def calculate_style_loss(self, generated_features, style_features):
        """
        Calculate the style loss as the L1 distance between Gram matrices.
        """
        loss = 0
        for gen_feat_name in self.style_layers_default:
            gen_feat = generated_features[gen_feat_name]
            sty_feat = style_features[gen_feat_name]
            loss += F.l1_loss(self.compute_gram_matrix(gen_feat), self.compute_gram_matrix(sty_feat))
        return loss

    def calculate_content_loss(self, generated_features, content_features):
        """
        Calculate the content loss as the L1 distance between feature maps.
        """
        loss = 0
        for content_feat_name in self.content_layers_default:
             loss += F.l1_loss(generated_features[content_feat_name], content_features[content_feat_name])
        return loss
    
    def forward(self, generated, real_style, real_content):
        # Normalize all images before feeding them to VGG
        gen_norm = self._normalize(generated)
        style_norm = self._normalize(real_style)
        content_norm = self._normalize(real_content)

        # Extract features for all three images
        all_layers = self.style_layers_default + self.content_layers_default
        gen_features = self.get_features(gen_norm, all_layers)
        style_features = self.get_features(style_norm, self.style_layers_default)
        content_features = self.get_features(content_norm, self.content_layers_default)
        
        # Calculate losses
        style_loss = self.calculate_style_loss(gen_features, style_features)
        content_loss = self.calculate_content_loss(gen_features, content_features)
        
        return content_loss, style_loss