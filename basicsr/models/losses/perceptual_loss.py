"""
Perceptual loss using VGG features
Based on EnlightenGAN and other perceptual loss implementations
"""
import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG19.
    
    Args:
        layer_weights (dict): Layer name and weight pairs for perceptual loss.
        vgg_type (str): Type of VGG network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize input images according to ImageNet stats.
        criterion (str): Criterion type. Default: 'l1'.
        perceptual_weight (float): Weight for perceptual loss. Default: 1.0.
        style_weight (float): Weight for style loss. Default: 0.0.
    """
    
    def __init__(self,
                 layer_weights={'conv3_4': 1.0, 'conv4_4': 1.0, 'conv5_4': 1.0},
                 vgg_type='vgg19',
                 use_input_norm=True,
                 criterion='l1',
                 perceptual_weight=1.0,
                 style_weight=0.0):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        
        # Load VGG network
        if vgg_type == 'vgg19':
            self.vgg = models.vgg19(pretrained=True).features
        else:
            raise NotImplementedError(f'VGG type {vgg_type} not supported.')
        
        # No need to train VGG
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer name mapping for VGG19
        self.layer_name_mapping = {
            '3': 'relu1_2',    # after conv1_2
            '8': 'relu2_2',    # after conv2_2
            '13': 'relu3_2',   # after conv3_2
            '15': 'conv3_4',   # after conv3_4
            '22': 'relu4_2',   # after conv4_2
            '24': 'conv4_4',   # after conv4_4
            '31': 'relu5_2',   # after conv5_2
            '33': 'conv5_4',   # after conv5_4
        }
        
        # ImageNet normalization
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        
        # Criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'Criterion {criterion} not supported.')
    
    def forward(self, x, gt):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        
        Returns:
            Tensor: Perceptual loss.
        """
        # Normalize input
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            gt = (gt - self.mean) / self.std
        
        # Extract features
        x_features = self._extract_features(x)
        gt_features = self._extract_features(gt)
        
        # Calculate perceptual loss
        percep_loss = 0
        for k in x_features.keys():
            if k in self.layer_weights:
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        
        return percep_loss * self.perceptual_weight
    
    def _extract_features(self, x):
        """Extract VGG features.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            dict: VGG features with layer names as keys.
        """
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layer_weights:
                    features[layer_name] = x.clone()
        return features
