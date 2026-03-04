"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
This module implements the SegFormer architecture for semantic segmentation using transformers.
Based on NVIDIA's implementation with modifications for efficient processing.
"""

# -*- coding: utf-8 -*-
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.segformer_backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for linear embedding.
    Projects input features to a different dimensional space.
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        """
        Initialize MLP module.
        
        Args:
            input_dim (int): Dimension of input features
            embed_dim (int): Dimension of embedded features
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass of MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Embedded tensor of shape (B, H*W, embed_dim)
        """
        x = x.flatten(2).transpose(1, 2)  # Reshape to (B, H*W, C)
        x = self.proj(x)  # Project to (B, H*W, embed_dim)
        return x
    

class ConvModule(nn.Module):
    """
    Convolution module with batch normalization and activation.
    Provides a standard conv-bn-act building block.
    """
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        """
        Initialize convolution module.
        
        Args:
            c1 (int): Number of input channels
            c2 (int): Number of output channels
            k (int): Kernel size
            s (int): Stride
            p (int): Padding
            g (int): Groups for grouped convolution
            act (bool or nn.Module): Activation function (True for ReLU)
        """
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """
        Forward pass of convolution module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after conv-bn-act
        """
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """
        Forward pass without batch normalization (for inference optimization).
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after conv-act
        """
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer decoder head for semantic segmentation.
    Combines multi-level features using MLP and interpolation.
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        """
        Initialize SegFormer decoder head.
        
        Args:
            num_classes (int): Number of segmentation classes
            in_channels (list): List of input channel dimensions for each level
            embedding_dim (int): Embedding dimension for MLP projection
            dropout_ratio (float): Dropout ratio for regularization
        """
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # MLP layers for each feature level
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # Fusion layer to combine all levels
        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,  # 4 levels combined
            c2=embedding_dim,
            k=1,
        )

        # Prediction and regularization layers
        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        """
        Forward pass of SegFormer decoder head.
        
        Args:
            inputs (list): List of feature maps from different levels [C1, C2, C3, C4]
            
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W)
        """
        c1, c2, c3, c4 = inputs

        # Process each level with MLP and interpolate to same resolution
        n, _, h, w = c4.shape
        
        # Process level 4 features
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Process level 3 features
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Process level 2 features
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # Process level 1 features
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Fuse all levels and make final prediction
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
    Combines transformer backbone with MLP decoder head.
    """
    def __init__(self, num_classes = 21, backbone = 'b0', pretrained = False):
        """
        Initialize SegFormer model.
        
        Args:
            num_classes (int): Number of segmentation classes
            backbone (str): Backbone type ('b0' to 'b5')
            pretrained (bool): Whether to use pretrained weights
        """
        super(SegFormer, self).__init__()
        
        # Define input channels for each backbone variant
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[backbone]
        
        # Initialize backbone based on selected variant
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[backbone](pretrained)
        
        # Define embedding dimension for each backbone variant
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[backbone]
        
        # Initialize decoder head
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        """
        Forward pass of SegFormer.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W)
        """
        H, W = inputs.size(2), inputs.size(3)
        
        # Extract features using backbone
        x = self.backbone.forward(inputs)
        
        # Decode features using SegFormer head
        x = self.decode_head.forward(x)
        
        # Interpolate to original input size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x