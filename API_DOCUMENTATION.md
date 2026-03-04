
# LEGDeeplab API Documentation

## Overview

LEGDeeplab (Lightweight Edge-Guided DeepLabv3+) is an advanced semantic segmentation framework that combines the DeepLabv3+ architecture with innovative attention mechanisms and edge guidance for superior performance with computational efficiency. The model is specifically designed to address the challenges in plant disease segmentation, such as handling small lesions and complex backgrounds. It has been successfully applied to the task of maize leaf rust detection.

## Key Innovations
1. **Large Selective Kernel (LSK) Block**: Dynamic receptive field adaptation based on input characteristics.
2. **Sparse Convolution (ScConv)**: Efficient feature processing through sparse connectivity with adaptive gating.
3. **Adaptive Graph Convolution Attention (AGCA)**: Captures long-range dependencies through dynamic graph construction and convolution.
4. **Edge Feature Aggregation (EFA)**: Enhances edge features for boundary preservation and better accuracy.

## Model Classes

### LEGDeeplab
The main LEGDeeplab model combining DeepLabv3+ with advanced attention mechanisms for improved performance on plant disease segmentation tasks.

#### Methods

##### `__init__(in_channels=3, num_classes=21, backbone="resnet18")`
Initialize the LEGDeeplab model, optimized for lightweight deployment on edge devices.

**Parameters:**
- `in_channels` (int): Number of input channels (default: 3 for RGB).
- `num_classes` (int): Number of segmentation classes (default: 21).
- `backbone` (str): Backbone network architecture (default: "resnet18" for computational efficiency).

##### `forward(x)`
Forward pass of the model for semantic segmentation.

**Parameters:**
- `x` (Tensor): Input tensor of shape (B, 3, H, W).

##### `freeze_backbone()`
Freeze backbone parameters for transfer learning.

##### `unfreeze_backbone()`
Unfreeze backbone parameters for fine-tuning.

## Training Pipeline
- **Preprocessing**: Handles multi-scale data augmentation to improve model robustness under real-world conditions.
- **Training Phases**: Freeze and unfreeze phases to optimize model performance while managing computational cost.

### Evaluation Pipeline
Evaluates model performance based on mIoU, precision, recall, and other relevant metrics.
