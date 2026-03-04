# LEGDeeplab API Documentation

## Overview

LEGDeeplab (Lightweight Edge-Guided DeepLabv3+) is an advanced semantic segmentation framework that combines the DeepLabv3+ architecture with innovative attention mechanisms and edge guidance for superior performance with computational efficiency.

## Table of Contents

1. [Architecture Components](#architecture-components)
2. [Model Classes](#model-classes)
3. [Utility Functions](#utility-functions)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation Pipeline](#evaluation-pipeline)
6. [Configuration Options](#configuration-options)

## Architecture Components

### LSKblock (Large Selective Kernel Block)

The LSKblock implements adaptive receptive field selection, allowing the network to dynamically adjust its focus based on input characteristics.

#### Methods
- `__init__(dim)`: Initialize LSK block with input/output channel dimension
- `forward(x)`: Process input tensor through LSK mechanism

#### Parameters
- `dim`: Input/output channel dimension

#### Returns
- Output tensor with same shape as input

### ScConv (Sparse Convolution)

The ScConv module implements efficient feature processing through sparse convolution with adaptive gating.

#### Methods
- `__init__(dim, expansion_ratio=2, kernel_size=7)`: Initialize ScConv module
- `forward(x)`: Process input through sparse convolution

#### Parameters
- `dim`: Input/output channel dimension
- `expansion_ratio`: Expansion ratio for intermediate features (default: 2)
- `kernel_size`: Kernel size for depthwise convolution (default: 7)

### AGCA (Adaptive Graph Convolution Attention)

Captures long-range spatial relationships through dynamic graph construction and convolution.

#### Methods
- `__init__(dim, num_heads=8)`: Initialize AGCA module
- `forward(x)`: Process input through graph attention mechanism

#### Parameters
- `dim`: Input feature dimension
- `num_heads`: Number of attention heads (default: 8)

### EFA (Edge Feature Aggregation)

Enhances edge features by aggregating multi-scale edge information for better boundary preservation.

#### Methods
- `__init__(dim)`: Initialize EFA module
- `forward(x)`: Process input through edge aggregation

#### Parameters
- `dim`: Input feature dimension

## Model Classes

### LEGDeeplab

The main LEGDeeplab model combining DeepLabv3+ with advanced attention mechanisms.

#### Methods

##### `__init__(in_channels=3, num_classes=21, backbone="resnet50")`
Initialize the LEGDeeplab model.

**Parameters:**
- `in_channels` (int): Number of input channels (default: 3 for RGB)
- `num_classes` (int): Number of segmentation classes (default: 21)
- `backbone` (str): Backbone network architecture (default: "resnet50")

**Raises:**
- `ValueError`: If unsupported backbone is provided

##### `forward(x)`
Forward pass of the model.

**Parameters:**
- `x` (Tensor): Input tensor of shape (B, 3, H, W)

**Returns:**
- Output tensor of shape (B, num_classes, H, W)

##### `freeze_backbone()`
Freeze backbone parameters for transfer learning.

##### `unfreeze_backbone()`
Unfreeze backbone parameters for fine-tuning.

## Utility Functions

### Training Utilities

#### `fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)`
Perform one training epoch.

#### `get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters)`
Get learning rate scheduler function.

#### `set_optimizer_lr(optimizer, lr_scheduler_func, epoch)`
Set learning rate for optimizer.

#### `weights_init(module)`
Initialize model weights.

### Data Loading Utilities

#### `unet_dataset_collate(batch)`
Collate function for DataLoader.

#### `resize_image(image, size)`
Resize image to specified size.

#### `preprocess_input(image)`
Preprocess input image for model.

### Evaluation Utilities

#### `compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)`
Compute mean Intersection over Union.

#### `evalute(y_true, y_pred, num_classes)`
Evaluate model predictions.

## Training Pipeline

### Command Line Arguments

The training script accepts various command line arguments for flexible configuration:

```bash
python train.py --model_name LEGDeeplab --backbone resnet50 --num_classes 21 \
                --input_shape 512 512 --batch_size 8 --epochs 200 \
                --learning_rate 0.0001 --fp16 --dice_loss --focal_loss --boundary_loss
```

### Training Phases

1. **Initialization Phase**: Setup model, optimizer, and data loaders
2. **Freeze Phase**: Train only decoder while freezing backbone (if enabled)
3. **Unfreeze Phase**: Train entire network with unfrozen backbone

### Loss Functions

The model supports multiple loss functions:

- **Cross Entropy Loss**: Standard classification loss
- **Dice Loss**: For handling class imbalance
- **Focal Loss**: For handling hard examples
- **Boundary Loss**: For improving boundary precision

## Evaluation Pipeline

### Evaluation Modes

The evaluation script supports three modes:

- **Mode 0**: Generate predictions and calculate mIoU
- **Mode 1**: Generate predictions only
- **Mode 2**: Calculate metrics only

### Evaluation Metrics

The evaluation provides comprehensive metrics:

- **mIoU**: Mean Intersection over Union
- **mPA**: Mean Pixel Accuracy
- **FWIoU**: Frequency Weighted IoU
- **Precision/Recall**: Per-class metrics
- **F1-Score**: Harmonic mean of precision and recall

## Configuration Options

### Model Configuration

- `model_name`: Model architecture ("LEGDeeplab")
- `backbone`: Backbone network ("resnet50", "resnet101", etc.)
- `num_classes`: Number of segmentation classes
- `input_shape`: Input image dimensions [height, width]

### Training Configuration

- `epochs`: Total number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Initial learning rate
- `min_lr`: Minimum learning rate
- `fp16`: Enable mixed precision training
- `dice_loss`: Enable Dice loss component
- `focal_loss`: Enable Focal loss component
- `boundary_loss`: Enable Boundary loss component

### Hardware Configuration

- `cuda`: Use CUDA acceleration
- `distributed`: Enable distributed training
- `sync_bn`: Use synchronized batch normalization
- `num_workers`: Number of data loading workers

## Advanced Features

### LEGDeeplab Innovations

1. **Large Selective Kernel (LSK)**: Adaptive receptive field selection
2. **Sparse Convolution (ScConv)**: Efficient feature processing
3. **Adaptive Graph Convolution Attention (AGCA)**: Long-range dependency modeling
4. **Edge Feature Aggregation (EFA)**: Boundary refinement

### Performance Optimizations

- Mixed precision training for speed and memory efficiency
- Distributed training support for multi-GPU setups
- Gradient checkpointing for memory-constrained environments
- Dynamic batch sizing based on available memory

## Error Handling

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **NaN Loss**: Check learning rate and data preprocessing
3. **Poor Convergence**: Verify data augmentation and learning rate schedule
4. **Model Not Learning**: Check class weights and loss function configuration

### Validation Checks

The framework includes validation checks for:
- Input tensor dimensions
- Class label ranges
- Model initialization
- Data loader configuration
- Hardware availability

## Best Practices

### For Training
- Start with frozen backbone for transfer learning
- Use mixed precision training for speed
- Monitor training metrics with TensorBoard
- Save checkpoints regularly

### For Evaluation
- Use appropriate evaluation metrics for your task
- Consider class imbalance in evaluation
- Validate on diverse test sets
- Compare against baseline methods

### For Deployment
- Export models to efficient formats (ONNX, TensorRT)
- Profile inference time and memory usage
- Consider quantization for edge devices
- Validate numerical precision after conversion

## References

For implementation details and theoretical background, refer to the original papers on:
- DeepLabv3+ architecture
- Attention mechanisms in computer vision
- Semantic segmentation benchmarks