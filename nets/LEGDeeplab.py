"""
LEGDeeplab: Lightweight Edge-Guided DeepLabv3+ for semantic segmentation.
This module implements a lightweight DeepLabv3+ variant with edge guidance and attention mechanisms.
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class LSKblock(nn.Module):
    """
    Large Selective Kernel (LSK) attention block.
    This module implements a spatial attention mechanism with large kernel convolutions.
    """
    def __init__(self, dim):
        """
        Initialize LSK block.
        
        Args:
            dim (int): Input feature dimension
        """
        super().__init__()
        # Depthwise convolutions with different kernel sizes
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        # Channel reduction and fusion
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        Forward pass of LSK block.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            torch.Tensor: Attention-weighted feature map
        """
        # Generate attention maps with different receptive fields
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        # Reduce channel dimensions
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # Combine and aggregate attention maps
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        # Apply attention weights and fuse
        attn = attn1 * sig[:, 0:1] + attn2 * sig[:, 1:2]
        attn = self.conv(attn)
        return x * attn


class ScConv(nn.Module):
    """
    Sparse Convolution (ScConv) block for efficient feature processing.
    This module implements a gating mechanism to reduce redundant computations.
    """
    def __init__(self, op_channel, group_num=4, gate_treshold=0.5):
        """
        Initialize ScConv block.
        
        Args:
            op_channel (int): Number of output channels
            group_num (int): Number of groups for GroupNorm
            gate_treshold (float): Threshold for gating mechanism
        """
        super().__init__()
        self.norm = nn.GroupNorm(group_num, op_channel)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of ScConv block.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            torch.Tensor: Processed feature map with sparse connections
        """
        # Normalize input features
        gn_x = self.norm(x)
        
        # Calculate adaptive weights based on normalization
        w_gamma = self.norm.weight / torch.sum(self.norm.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)

        # Separate informative and non-informative features
        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold

        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x

        # Cross-connection for feature enhancement
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)

        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class AGCA(nn.Module):
    """
    Adaptive Graph Convolution Attention (AGCA) module.
    This module implements graph-based attention for global context modeling.
    """

    def __init__(self, in_channel, ratio=4):
        """
        Initialize AGCA module.
        
        Args:
            in_channel (int): Number of input channels
            ratio (int): Reduction ratio for hidden channels
        """
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio

        # Global context extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        # Graph attention parameters
        self.register_buffer('A0', torch.eye(hide_channel))  # Identity matrix for base graph
        self.A2 = nn.Parameter(torch.zeros((hide_channel, hide_channel)))
        nn.init.constant_(self.A2, 1e-6)

        # Graph convolution operations
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of AGCA module.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            torch.Tensor: Attention-weighted feature map
        """
        # Ensure device compatibility
        if self.A0.device != x.device:
            self.A0 = self.A0.to(x.device)

        # Extract global context
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)

        # Graph attention computation
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2

        # Graph convolution and feature enhancement
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))
        return x * y


class EFA(nn.Module):
    """
    Edge Feature Aggregation (EFA) module.
    This module focuses on edge feature extraction and enhancement.
    """
    def __init__(self, in_channels):
        super(EFA, self).__init__()
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(1, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, edge, x):
        edge_att = F.interpolate(edge, size=x.shape[2:], mode='bilinear', align_corners=True)
        edge_weight = self.edge_enhance(edge_att)
        detail_weight = self.detail_enhance(x)
        attn = edge_weight * detail_weight
        return x * attn + x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, dilation, dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels, 1, dilation[0])
        self.conv3 = DepthwiseSeparableConv(in_channels, out_channels, 1, dilation[1])
        self.conv4 = DepthwiseSeparableConv(in_channels, out_channels, 1, dilation[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.agca = AGCA(out_channels * (len(dilation) + 2))

    def forward(self, x):
        size = x.shape[-2:]

        x1 = self.conv1x1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_pool = self.pool(x)
        x_pool = F.interpolate(x_pool, size, mode='bilinear', align_corners=False)
        x_pool = self.conv1x1(x_pool)

        x_cat = torch.cat([x1, x2, x3, x4, x_pool], dim=1)
        x_cat = self.agca(x_cat)
        return self.project(x_cat)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, use_agca=False, agca_ratio=8):
        super().__init__()
        self.main_path = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, stride, dilation)
        )

        if out_channels in [48, 64, 384, 512]:
            self.main_path.add_module("scconv", ScConv(out_channels))
        elif out_channels in [96, 128, 192, 256]:
            self.main_path.add_module("lskblock", LSKblock(out_channels))

        self.residual_path = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.residual_path.add_module(
                "conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            self.residual_path.add_module("bn", nn.BatchNorm2d(out_channels))

        if use_agca:
            self.residual_path.add_module("agca", AGCA(out_channels, ratio=agca_ratio))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_path(x)
        x = self.main_path(x)
        x = x + residual
        return self.relu(x)



class LEGDeeplab(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = EdgeAwareEncoder(in_channels)
        self.decoder = AdaptiveDecoder(64, num_classes)

    def forward(self, x):
        edge, low, high = self.encoder(x)
        return self.decoder(edge, low, high)


class EdgeAwareEncoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(32, 48, 1, dilation=1),
            ResidualBlock(48, 64, 1, dilation=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 96, 2, dilation=1),
            ResidualBlock(96, 128, 1, dilation=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 192, 2, dilation=1),
            ResidualBlock(192, 256, 1, dilation=1)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 384, 1, dilation=2),
            ResidualBlock(384, 512, 1, dilation=2)
        )

        self.aspp = ASPP(512, 64, [6, 12, 18])

    def forward(self, x):
        B, C, H, W = x.shape
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        sobel_x = self.sobel_x.to(x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(x.device, dtype=x.dtype)
        grad_x = F.conv2d(gray, sobel_x.repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(gray, sobel_y.repeat(1, 1, 1, 1), padding=1)

        
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        edge_flat = edge.view(B, -1)
        percentile_95 = torch.quantile(edge_flat, 0.95, dim=1, keepdim=True).view(B, 1, 1, 1)
        
        min_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        edge_clipped = torch.clamp(edge, min_val, percentile_95)
       
        edge_flat_clipped = edge_clipped.view(B, -1)
        edge_min = edge_flat_clipped.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_max = edge_flat_clipped.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
       
        edge_max = torch.clamp(edge_max - edge_min, min=1e-8)
        edge_normalized = (edge_clipped - edge_min) / edge_max * 255.0

        edge_int = torch.clamp(edge_normalized, 0.0, 255.0).to(torch.uint8)
        edge_float = edge_int.float()

        hist = []
        for b in range(B):
            hist_b = torch.histc(edge_float[b], bins=256, min=0, max=255)
            hist.append(hist_b)
        hist = torch.stack(hist)

        cdf = hist.cumsum(dim=1)
        cdf_min = cdf[:, 0].view(B, 1)
        total_pixels = H * W
        cdf_normalized = (cdf - cdf_min) / torch.clamp(total_pixels - cdf_min, min=1e-8)

        edge_eq = []
        for b in range(B):
            eq_b = cdf_normalized[b, edge_int[b].long()] * 255.0
            edge_eq.append(eq_b)
        edge_eq = torch.stack(edge_eq).float()

        edge = edge_eq / 255.0

        x = self.backbone(x)
        low = self.layer1(x)  
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        high = self.aspp(x)

        return edge, low, high


class AdaptiveDecoder(nn.Module):

    def __init__(self, low_channels, num_classes):
        super().__init__()
        self.upsample_high = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Sequential(
            DepthwiseSeparableConv(low_channels + 64, 64, 1, 1),
            DepthwiseSeparableConv(64, 64, 1, 1)
        )

        self.EFA = EFA(64)

        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            LSKblock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ScConv(64)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, edge, low_feature, high_feature):
        high_feature = self.upsample_high(high_feature)
        x = torch.cat([low_feature, high_feature], dim=1)
        x = self.fusion_conv(x)
        x = self.EFA(edge, x)
        x = self.upsample_block(x)
        return self.final_conv(x)
