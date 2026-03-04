"""
LEGDeeplab Model Configuration
=============================

This configuration file defines the LEGDeeplab architecture with advanced attention mechanisms
and edge-guidance features for semantic segmentation.

Architecture Details:
- Encoder-Decoder with atrous convolutions (DeepLabv3+ foundation)
- Large Selective Kernel (LSK) blocks for adaptive receptive fields
- Sparse Convolution (ScConv) for efficient feature processing
- Adaptive Graph Convolution Attention (AGCA) for long-range dependencies
- Edge Feature Aggregation (EFA) for boundary refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class LSKblock(nn.Module):
    """
    Large Selective Kernel Block for adaptive receptive field selection.
    
    This module implements the Large Selective Kernel mechanism that adaptively selects
    receptive fields based on input characteristics, allowing the network to focus on
    different scales of features dynamically.
    
    Features:
    - Multi-branch convolutions with different kernel sizes
    - Attention-based feature fusion
    - Channel-wise and spatial attention mechanisms
    """
    
    def __init__(self, dim):
        """
        Initialize LSK block.
        
        Args:
            dim: Input/output channel dimension
        """
        super(LSKblock, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(dim//2, dim//2, 3, padding=1, groups=dim//2),
            nn.Conv2d(dim//2, dim//2, 3, padding=1, groups=dim//2)
        )
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of LSK block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        att = self.conv0(x)
        att = self.conv_spatial(att)
        
        out1 = self.conv1(att)
        out2 = self.conv2(att)
        
        out1 = self.act(out1)
        out1 = self.conv_seq(out1)
        
        out2 = self.act(out2)
        out2 = self.conv_seq(out2)
        
        out = torch.cat([out1, out2], dim=1)
        out = self.sigmoid(out)
        
        return out * x


class ScConv(nn.Module):
    """
    Sparse Convolution Module for efficient feature processing.
    
    This module implements a sparse convolution approach that reduces computational
    overhead while preserving important feature information through adaptive gating.
    
    Features:
    - Group convolution for efficiency
    - Channel shuffling for cross-group information exchange
    - Squeeze-and-excitation for channel attention
    """
    
    def __init__(self, dim, expansion_ratio=2, kernel_size=7):
        """
        Initialize ScConv module.
        
        Args:
            dim: Input/output channel dimension
            expansion_ratio: Expansion ratio for intermediate features
            kernel_size: Kernel size for depthwise convolution
        """
        super(ScConv, self).__init__()
        med_ch = int(dim * expansion_ratio)
        self.pw1 = nn.Conv2d(dim, med_ch, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(med_ch)
        
        self.dw_conv = nn.Conv2d(med_ch, med_ch, kernel_size, 1, kernel_size//2, groups=med_ch, bias=False)
        self.norm2 = nn.BatchNorm2d(med_ch)
        
        self.sc_conv = nn.Conv2d(med_ch//2, med_ch//2, kernel_size, 1, kernel_size//2, groups=med_ch//2, bias=False)
        
        self.pw2 = nn.Conv2d(med_ch, dim, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(dim)
        
        self.act = nn.ReLU(inplace=True)
        self.se = SEModule(dim)

    def forward(self, x):
        """
        Forward pass of ScConv module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.dw_conv(x)
        x = self.norm2(x)
        x = self.act(x)
        
        # Channel shuffling
        B, C, H, W = x.shape
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.sc_conv(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = self.pw2(x)
        x = self.norm3(x)
        
        x = self.se(x)
        
        return x + residual


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module for channel attention.
    
    This module implements channel-wise attention by learning to recalibrate
    channel-wise feature responses.
    """
    
    def __init__(self, channels, reduction=16):
        """
        Initialize SE module.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of SE module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class AGCA(nn.Module):
    """
    Adaptive Graph Convolution Attention for long-range dependencies.
    
    This module constructs dynamic graphs based on feature similarity and performs
    graph convolution to capture long-range spatial relationships.
    """
    
    def __init__(self, dim, num_heads=8):
        """
        Initialize AGCA module.
        
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
        """
        super(AGCA, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Graph construction parameters
        self.graph_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Forward pass of AGCA module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # Generate Q, K, V
        qkv = self.qkv(x_permuted).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        
        # Apply attention
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x_attn = x_attn.permute(0, 3, 1, 2).contiguous()
        
        # Graph convolution
        x_graph = self.graph_conv(x)
        
        # Combine attention and graph features
        output = self.proj((x_attn + x_graph).permute(0, 2, 3, 1).contiguous().view(B, H * W, C))
        output = output.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return output


class EFA(nn.Module):
    """
    Edge Feature Aggregation for boundary refinement.
    
    This module enhances edge features by aggregating multi-scale edge information
    and integrating it with semantic features for better boundary preservation.
    """
    
    def __init__(self, dim):
        """
        Initialize EFA module.
        
        Args:
            dim: Input feature dimension
        """
        super(EFA, self).__init__()
        
        # Edge detection kernels
        self.edge_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, dim * 4 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 4 // 16, dim * 4, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(dim * 4, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        """
        Forward pass of EFA module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Extract multi-scale features
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        
        # Extract edge features
        edges = self.edge_conv(x)
        
        # Concatenate all features
        concat_feat = torch.cat([feat1, feat3, feat5, edges], dim=1)
        
        # Apply attention
        attention_weights = self.attention(concat_feat)
        weighted_feat = concat_feat * attention_weights
        
        # Fuse features
        output = self.fusion(weighted_feat)
        output = self.norm(output)
        
        return output + x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for multi-scale feature extraction.
    
    This module implements the ASPP component of DeepLabv3+, capturing features
    at multiple scales using dilated convolutions.
    """
    
    def __init__(self, in_channels, out_channels, atrous_rates):
        """
        Initialize ASPP module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            atrous_rates: List of dilation rates for atrous convolutions
        """
        super(ASPP, self).__init__()
        
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for rate in atrous_rates:
            modules.append(
                ASPPConv(in_channels, out_channels, rate)
            )
        
        modules.append(
            ASPPPooling(in_channels, out_channels)
        )
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of ASPP module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    """Atrous convolution block for ASPP."""
    
    def __init__(self, in_channels, out_channels, dilation):
        """
        Initialize ASPP convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation: Dilation rate for convolution
        """
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Sequential):
    """ASPP pooling block for global context aggregation."""
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize ASPP pooling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of ASPP pooling block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Decoder(nn.Module):
    """
    Decoder module for upsampling and feature refinement.
    
    This module implements the decoder component of DeepLabv3+, combining
    low-level features with high-level semantic features for precise localization.
    """
    
    def __init__(self, num_classes, backbone, low_level_channels=256):
        """
        Initialize decoder module.
        
        Args:
            num_classes: Number of segmentation classes
            backbone: Backbone network identifier
            low_level_channels: Number of channels in low-level features
        """
        super(Decoder, self).__init__()
        
        if backbone == 'resnet50':
            project_channels = 48  # Low-level feature channels from ResNet-50
        else:
            project_channels = 48  # Default value
        
        self.conv1 = nn.Conv2d(low_level_channels, project_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(project_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Additional convolution blocks
        self.last_conv = nn.Sequential(
            nn.Conv2d(256 + project_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        
        # LEGDeeplab specific enhancements
        self.lsk_block = LSKblock(256 + project_channels)
        self.sc_conv = ScConv(256 + project_channels)
        self.agca = AGCA(256 + project_channels)
        self.efa = EFA(256 + project_channels)

    def forward(self, x, low_level_feat):
        """
        Forward pass of decoder module.
        
        Args:
            x: High-level features from ASPP
            low_level_feat: Low-level features from backbone
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x, low_level_feat), dim=1)
        
        # Apply LEGDeeplab enhancements
        x = self.lsk_block(x)
        x = self.sc_conv(x)
        x = self.agca(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # AGCA expects NCHW->NHWC->NCHW
        x = self.efa(x)
        
        x = self.last_conv(x)
        
        return x


class LEGDeeplab(nn.Module):
    """
    LEGDeeplab: Lightweight Edge-Guided DeepLabv3+ for Advanced Semantic Segmentation.
    
    This architecture combines the DeepLabv3+ framework with innovative attention
    mechanisms and edge guidance for superior segmentation performance with
    computational efficiency.
    
    Key Innovations:
    1. Large Selective Kernel (LSK) blocks for adaptive receptive fields
    2. Sparse Convolution (ScConv) for efficient processing
    3. Adaptive Graph Convolution Attention (AGCA) for long-range dependencies
    4. Edge Feature Aggregation (EFA) for boundary refinement
    """
    
    def __init__(self, in_channels=3, num_classes=21, backbone="resnet50"):
        """
        Initialize LEGDeeplab model.
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            num_classes: Number of segmentation classes
            backbone: Backbone network architecture
        """
        super(LEGDeeplab, self).__init__()
        
        if backbone == "resnet50":
            from torchvision.models import resnet50
            from torchvision.models._utils import IntermediateLayerGetter
            
            # Load pretrained ResNet50 backbone
            backbone_model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            
            # Remove fully connected layer
            backbone_layers = list(backbone_model.children())[:-2]
            self.backbone = nn.Sequential(*backbone_layers)
            
            # Define return layers for feature extraction
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
            self.feature_extractor = IntermediateLayerGetter(self.backbone, return_layers)
            
            # ASPP module configuration
            self.aspp = ASPP(
                in_channels=2048,  # ResNet50 layer4 output channels
                out_channels=256,
                atrous_rates=[12, 24, 36]
            )
            
            # Decoder
            self.decoder = Decoder(num_classes, backbone)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # LEGDeeplab specific enhancements in decoder
        self.num_classes = num_classes
        self.backbone_name = backbone

    def forward(self, x):
        """
        Forward pass of LEGDeeplab model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        # Extract features from backbone
        features = self.feature_extractor(x)
        x = features['out']  # High-level features
        low_level_feat = features['low_level']  # Low-level features
        
        # Apply ASPP for multi-scale feature extraction
        x = self.aspp(x)
        
        # Decode features with LEGDeeplab enhancements
        x = self.decoder(x, low_level_feat)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[0], self.num_classes, x.shape[2]*4, x.shape[3]*4), 
                         mode='bilinear', align_corners=False)
        
        return x

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True