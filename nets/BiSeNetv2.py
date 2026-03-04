"""
BiSeNetV2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
This module implements the BiSeNetV2 architecture which combines detail and semantic branches
for efficient real-time semantic segmentation.
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class StemBlock(nn.Module):
    """
    Stem block for initial feature extraction.
    Processes input image with two parallel paths: convolution and pooling.
    """
    def __init__(self, in_channels=3, out_channels=16):
        """
        Initialize stem block.
        
        Args:
            in_channels (int): Number of input channels (typically 3 for RGB)
            out_channels (int): Number of output channels
        """
        super(StemBlock, self).__init__()

        # Initial convolution path
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Branch convolution path with reduction
        self.conv_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Max pooling path
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of stem block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H/4, W/4)
        """
        x = self.conv_in(x)

        x_branch = self.conv_branch(x)
        x_downsample = self.pool(x)
        out = torch.cat([x_branch, x_downsample], dim=1)
        out = self.fusion(out)

        return out


class depthwise_separable_conv(nn.Module):
    """
    Depthwise separable convolution for efficient feature extraction.
    Separates spatial and channel-wise convolutions to reduce computational cost.
    """
    def __init__(self, in_channels, out_channels, stride):
        """
        Initialize depthwise separable convolution.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the depthwise convolution
        """
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of depthwise separable convolution.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after depthwise and pointwise convolutions
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class GELayer(nn.Module):
    """
    Geometry-Enhancement Layer (GE) for BiSeNetV2.
    Enhances geometric details while preserving semantic information.
    """
    def __init__(self, in_channels, out_channels, exp_ratio=6, stride=1):
        """
        Initialize GE layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            exp_ratio (int): Expansion ratio for intermediate channels
            stride (int): Stride for the layer (1 for same resolution, 2 for downsample)
        """
        super(GELayer, self).__init__()
        mid_channel = in_channels * exp_ratio
        
        # Initial convolution for channel mixing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Depthwise convolution branch
        if stride == 1:
            self.dwconv = nn.Sequential(
                # ReLU in ConvModule not shown in paper
                nn.Conv2d(in_channels, mid_channel, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),

                depthwise_separable_conv(mid_channel, mid_channel, stride=1),
                nn.BatchNorm2d(mid_channel),
            )
            self.shortcut = None
        else:
            # For stride=2, use additional convolutions for downsampling
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, 3, stride=1, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),

                # ReLU in ConvModule not shown in paper
                depthwise_separable_conv(mid_channel, mid_channel, stride=stride),
                nn.BatchNorm2d(mid_channel),

                depthwise_separable_conv(mid_channel, mid_channel, stride=1),
                nn.BatchNorm2d(mid_channel),
            )

            # Shortcut path for downsampling
            self.shortcut = nn.Sequential(
                depthwise_separable_conv(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),

                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        # Final projection to output channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of GE layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after GE processing
        """
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


class CEBlock(nn.Module):
    """
    Context Embedding Block (CE) for BiSeNetV2.
    Captures global context information and enhances spatial details.
    """
    def __init__(self, in_channels=16, out_channels=16):
        """
        Initialize context embedding block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(CEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Global average pooling for context extraction
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # AdaptiveAvgPool2d 把形状变为(Batch size, N, 1, 1)后，batch size=1不能正常通过BatchNorm2d， 但是batch size>1是可以正常通过的。如果想开启BatchNorm，训练时batch size>1即可，测试时使用model.eval()即不会报错。
            # nn.BatchNorm2d(self.in_channels)
        )

        # Channel transformation after GAP
        self.conv_gap = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0),
            # nn.BatchNorm2d(self.out_channels), 同上
            nn.ReLU()
        )

        # Note: in paper here is naive conv2d, no bn-relu
        self.conv_last = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        """
        Forward pass of context embedding block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with enhanced context
        """
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x


class DetailBranch(nn.Module):
    """
    Detail branch of BiSeNetV2.
    Preserves high-resolution spatial details for accurate boundary localization.
    """
    def __init__(self, detail_channels=(64, 64, 128), in_channels=3):
        """
        Initialize detail branch.
        
        Args:
            detail_channels (tuple): Channel dimensions for each stage
            in_channels (int): Number of input channels
        """
        super(DetailBranch, self).__init__()
        self.detail_branch = nn.ModuleList()

        for i in range(len(detail_channels)):
            if i == 0:
                # First stage: downsample input channels to first detail channel count
                self.detail_branch.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, detail_channels[i], 3, stride=2, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),
                    )
                )
            else:
                # Subsequent stages: downsample previous stage to current stage
                self.detail_branch.append(
                    nn.Sequential(
                        nn.Conv2d(detail_channels[i - 1], detail_channels[i], 3, stride=2, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU()
                    )
                )

    def forward(self, x):
        """
        Forward pass of detail branch.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: High-resolution feature map with preserved details
        """
        for stage in self.detail_branch:
            x = stage(x)
        return x


class SemanticBranch(nn.Module):
    """
    Semantic branch of BiSeNetV2.
    Captures rich semantic information with reduced spatial resolution for efficiency.
    """
    def __init__(self, semantic_channels=(16, 32, 64, 128), in_channels=3, exp_ratio=6):
        """
        Initialize semantic branch.
        
        Args:
            semantic_channels (tuple): Channel dimensions for each stage
            in_channels (int): Number of input channels
            exp_ratio (int): Expansion ratio for GE layers
        """
        super(SemanticBranch, self).__init__()
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = nn.ModuleList()

        for i in range(len(semantic_channels)):
            if i == 0:
                # First stage: use stem block to reduce resolution
                self.semantic_stages.append(StemBlock(self.in_channels, semantic_channels[i]))

            elif i == (len(semantic_channels) - 1):
                # Last stages: multiple GE layers with downsampling and identity
                self.semantic_stages.append(
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1),

                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1)
                    )
                )

            else:
                # Middle stages: GE layer with downsampling followed by identity
                self.semantic_stages.append(
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)
                    )
                )

        # Add context embedding block at the end
        self.semantic_stages.append(CEBlock(semantic_channels[-1], semantic_channels[-1]))

    def forward(self, x):
        """
        Forward pass of semantic branch.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: List of feature maps at different stages
        """
        semantic_outs = []
        for semantic_stage in self.semantic_stages:
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


class AggregationLayer(nn.Module):
    """
    Aggregation layer for BiSeNetV2.
    Fuses information from detail and semantic branches using guided aggregation.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize aggregation layer.
        
        Args:
            in_channels (int): Number of input channels from both branches
            out_channels (int): Number of output channels
        """
        super(AggregationLayer, self).__init__()
        
        # Processing for detail branch features
        self.Conv_DetailBranch_1 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.Conv_DetailBranch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Processing for semantic branch features
        self.Conv_SemanticBranch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.Conv_SemanticBranch_2 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Final output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, Detail_x, Semantic_x):
        """
        Forward pass of aggregation layer.
        
        Args:
            Detail_x (torch.Tensor): Feature map from detail branch
            Semantic_x (torch.Tensor): Feature map from semantic branch
            
        Returns:
            torch.Tensor: Fused feature map combining detail and semantic information
        """
        DetailBranch_1 = self.Conv_DetailBranch_1(Detail_x)
        DetailBranch_2 = self.Conv_DetailBranch_2(Detail_x)

        SemanticBranch_1 = self.Conv_SemanticBranch_1(Semantic_x)
        SemanticBranch_2 = self.Conv_SemanticBranch_2(Semantic_x)

        # Matrix multiplication for feature fusion (Note: this implementation may need adjustment)
        out_1 = torch.matmul(DetailBranch_1, SemanticBranch_1)
        out_2 = torch.matmul(DetailBranch_2, SemanticBranch_2)
        out_2 = F.interpolate(out_2, scale_factor=4, mode="bilinear", align_corners=True)

        out = torch.matmul(out_1, out_2)
        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    """
    Segmentation head for BiSeNetV2.
    Converts feature maps to class predictions.
    """
    def __init__(self, channels, num_classes):
        """
        Initialize segmentation head.
        
        Args:
            channels (int): Number of input feature channels
            num_classes (int): Number of segmentation classes
        """
        super().__init__()
        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, x):
        """
        Forward pass of segmentation head.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            torch.Tensor: Class prediction map
        """
        return self.cls_seg(x)


class BiSeNetV2(nn.Module):
    """
    BiSeNetV2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation.
    Combines detail and semantic branches for efficient and accurate segmentation.
    """
    def __init__(self, in_channels=3,
                 detail_channels=(64, 64, 128),
                 semantic_channels=(16, 32, 64, 128),
                 semantic_expansion_ratio=6,
                 aggregation_channels=128,
                 out_indices=(0, 1, 2, 3, 4),
                 num_classes=3):
        """
        Initialize BiSeNetV2 model.
        
        Args:
            in_channels (int): Number of input channels (typically 3 for RGB)
            detail_channels (tuple): Channel dimensions for detail branch stages
            semantic_channels (tuple): Channel dimensions for semantic branch stages
            semantic_expansion_ratio (int): Expansion ratio for GE layers in semantic branch
            aggregation_channels (int): Channels for aggregation layer
            out_indices (tuple): Indices of stages to output
            num_classes (int): Number of segmentation classes
        """
        super(BiSeNetV2, self).__init__()

        # Store configuration parameters
        self.in_channels = in_channels
        self.detail_channels = detail_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.semantic_channels = semantic_channels
        self.aggregation_channels = aggregation_channels
        self.out_indices = out_indices
        self.num_classes = num_classes

        # Initialize network branches
        self.detail = DetailBranch(detail_channels=self.detail_channels, in_channels=self.in_channels)
        self.semantic = SemanticBranch(semantic_channels=self.semantic_channels, in_channels=self.in_channels,
                                       exp_ratio=self.semantic_expansion_ratio)
        self.AggregationLayer = AggregationLayer(in_channels=self.aggregation_channels,
                                                 out_channels=self.aggregation_channels)

        # Initialize segmentation heads
        self.seg_head_aggre = SegHead(semantic_channels[-1], self.num_classes)
        self.seg_heads = nn.ModuleList()
        self.seg_heads.append(self.seg_head_aggre)
        for channel in semantic_channels:
            self.seg_heads.append(SegHead(channel, self.num_classes))

    def forward(self, x):
        """
        Forward pass of BiSeNetV2.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)
            
        Returns:
            list: List of segmentation outputs from different stages
        """
        _, _, h, w = x.size()
        
        # Process through detail branch
        x_detail = self.detail(x)
        
        # Process through semantic branch (returns list of features from different stages)
        x_semantic_lst = self.semantic(x)
        
        # Fuse detail and semantic features
        x_head = self.AggregationLayer(x_detail, x_semantic_lst[-1])
        
        # Prepare outputs from fused features and semantic features
        outs = [x_head] + x_semantic_lst[:-1]
        outs = [outs[i] for i in self.out_indices]

        out = tuple(outs)

        # Generate segmentation results for each output stage
        seg_out = []
        for index, stage in enumerate(self.seg_heads):
            seg_out.append(F.interpolate(stage(out[index]), size=(h, w), mode="bilinear", align_corners=True))
        return seg_out