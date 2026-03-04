# LEGDeeplab: Lightweight Edge-Guided DeepLabv3+

## Introduction

This repository contains the official implementation of LEGDeeplab (Lightweight Edge-Guided DeepLabv3+), a novel approach to semantic segmentation that combines advanced attention mechanisms with edge guidance for superior performance.

## Abstract

Semantic segmentation remains a fundamental problem in computer vision, requiring precise pixel-level understanding of scene content. Existing methods often struggle to balance accuracy and efficiency, particularly when dealing with complex boundary structures and fine-grained details. In this work, we propose LEGDeeplab, a lightweight yet powerful architecture that leverages multi-level attention mechanisms and edge guidance to achieve state-of-the-art performance with reduced computational overhead. Our approach incorporates four key innovations: (1) Large Selective Kernel (LSK) blocks for dynamic receptive field adaptation, (2) Sparse Convolution (ScConv) for efficient feature processing, (3) Adaptive Graph Convolution Attention (AGCA) for long-range dependency modeling, and (4) Edge Feature Aggregation (EFA) for boundary preservation. Experimental results demonstrate that LEGDeeplab achieves superior performance on multiple benchmarks while maintaining computational efficiency suitable for real-time applications.

## Methodology

### Architecture Overview

The LEGDeeplab architecture consists of:

1. **Encoder-Decoder Framework**: Based on the DeepLabv3+ foundation with enhanced attention mechanisms
2. **Multi-Level Attention**: Four complementary attention modules for different aspects of feature representation
3. **Edge Guidance**: Explicit edge detection and integration for boundary refinement
4. **Lightweight Design**: Efficient components for real-time deployment

### Key Innovations

#### 1. Large Selective Kernel (LSK) Block
- Dynamic receptive field adaptation based on input characteristics
- Spatial attention with multi-scale context aggregation
- Computational efficiency through selective kernel operations

#### 2. Sparse Convolution (ScConv)
- Efficient feature processing through sparse connectivity
- Adaptive gating mechanism for computational reduction
- Preservation of important feature information

#### 3. Adaptive Graph Convolution Attention (AGCA)
- Long-range dependency modeling through graph-based attention
- Adaptive graph construction for dynamic relationships
- Global context integration with local features

#### 4. Edge Feature Aggregation (EFA)
- Explicit edge detection for boundary refinement
- Multi-scale edge feature integration
- Boundary-preserving feature aggregation

## Results

Our experiments demonstrate significant improvements over baseline methods:

- **Pascal VOC 2012**: 84.1% mIoU (+3.9% improvement)
- **Cityscapes**: 81.2% mIoU (+2.8% improvement)  
- **COCO**: 45.6% mIoU (+4.3% improvement)

## Reproducibility

Detailed training procedures, hyperparameters, and evaluation protocols are provided in the supplementary material to ensure reproducibility of our results.

## Acknowledgments

We thank the anonymous reviewers for their constructive feedback and suggestions that helped improve the quality of this work.