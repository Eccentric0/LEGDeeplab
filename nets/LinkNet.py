"""
LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
This module implements the LinkNet architecture for semantic segmentation.
"""

import torch
from torch import nn

class LinkNet(nn.Module):
    """
    LinkNet implementation for semantic segmentation.
    LinkNet connects encoder and decoder layers directly to preserve spatial information.
    """
    def __init__(self, in_channels, num_classes) -> None:
        """
        Initialize LinkNet model.
        
        Args:
            in_channels (int): Number of input channels (typically 3 for RGB)
            num_classes (int): Number of segmentation classes
        """
        super().__init__()

        # Initial feature extraction layers
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),  # First convolution layer
            nn.BatchNorm2d(64),                               # Batch normalization
            nn.ReLU(inplace=True),                            # Activation function
            nn.MaxPool2d(3, 2, 1)                           # Max pooling
        )  # Output: 64 channels, 1/4 resolution

        # Encoder blocks with increasing channels
        self.encoder1 = encoder_block(64, 64)    # 64 channels, 1/8 resolution
        self.encoder2 = encoder_block(64, 128)   # 128 channels, 1/16 resolution
        self.encoder3 = encoder_block(128, 256)  # 256 channels, 1/32 resolution
        self.encoder4 = encoder_block(256, 512)  # 512 channels, 1/64 resolution

        # Decoder blocks with decreasing channels
        self.decoder4 = decoder_block(512, 256)  # 256 channels, 1/32 resolution
        self.decoder3 = decoder_block(256, 128)  # 128 channels, 1/16 resolution
        self.decoder2 = decoder_block(128, 64)   # 64 channels, 1/8 resolution
        self.decoder1 = decoder_block(64, 64)    # 64 channels, 1/4 resolution

        # Final classification head
        self.classifer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=False),  # Upsample by factor of 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),              # Regular convolution
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, num_classes, 2, 2)             # Final upsampling to original size
        )  # Output: num_classes channels, original resolution

    def forward(self, x):
        """
        Forward pass of LinkNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, num_classes, height, width)
        """
        # Extract initial features
        x = self.embedding(x)

        # Encode features through encoder blocks
        x1 = self.encoder1(x)   # First encoder output
        x2 = self.encoder2(x1)  # Second encoder output
        x3 = self.encoder3(x2)  # Third encoder output
        x4 = self.encoder4(x3)  # Fourth encoder output

        # Decode features through decoder blocks with skip connections
        x = self.decoder4(x4)           # Decode deepest features
        x = self.decoder3(x + x3)       # Skip connection from encoder3
        x = self.decoder2(x + x2)       # Skip connection from encoder2
        x = self.decoder1(x + x1)       # Skip connection from encoder1

        # Final classification
        x = self.classifer(x)
        return x


class encoder_block(nn.Module):
    """
    Encoder block for LinkNet.
    Contains two residual blocks with specified input and output channels.
    """
    def __init__(self, in_channels, out_channels) -> None:
        """
        Initialize encoder block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.conv1 = residual_block(in_channels, out_channels, 2, 1)  # Downsample by factor of 2
        self.conv2 = residual_block(out_channels, out_channels, 1, 1)  # Same resolution

    def forward(self, x):
        """
        Forward pass of encoder block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after processing
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class residual_block(nn.Module):
    """
    Residual block implementation for LinkNet.
    Implements the basic residual connection with optional downsampling.
    """
    def __init__(self, in_channels, out_channels, stride, dilation) -> None:
        """
        Initialize residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
            dilation (int): Dilation rate for the first convolution
        """
        super().__init__()

        # First convolutional layer with stride and dilation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.acti = nn.ReLU(inplace=True)

        # Shortcut connection to handle different channel numbers or strides
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass of residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after residual processing
        """
        # Apply shortcut connection
        short = self.shortcut(x)
        
        # Apply main convolutional path
        x = self.conv1(x)
        x = self.acti(x)
        x = self.conv2(x)
        
        # Add residual connection and apply activation
        x = self.acti(x + short)
        return x


class decoder_block(nn.Module):
    """
    Decoder block for LinkNet.
    Performs upsampling to increase spatial resolution.
    """
    def __init__(self, in_channels, out_channels) -> None:
        """
        Initialize decoder block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()

        # Calculate hidden channels for bottleneck design
        hidden_channels = in_channels // 4
        
        # Sequential layers for upsampling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),  # Bottleneck layer
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, 3, 2, 1, 1, bias=False),  # Upsample by 2x
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),  # Expand to output channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of decoder block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after upsampling
        """
        return self.conv(x)