"""
Utility functions for semantic segmentation pipeline.
This module contains image processing, configuration, and helper functions.
"""

import os
import random
import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    """
    Convert image to RGB format if not already in RGB.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: RGB formatted image
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    """
    Resize image while maintaining aspect ratio with padding.
    
    Args:
        image (PIL.Image): Input image
        size (tuple): Target size (width, height)
        
    Returns:
        tuple: (resized_image, new_width, new_height)
    """
    iw, ih  = image.size
    w, h    = size

    # Calculate scaling factor to maintain aspect ratio
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    # Resize image and create padded version
    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))  # Gray padding
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    

def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def seed_everything(seed=11):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """
    Initialize worker for data loading with proper seeding.
    
    Args:
        worker_id (int): Worker ID
        rank (int): Process rank
        seed (int): Base seed value
    """
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    image /= 255.0
    return image


def show_config(**kwargs):
    """
    Display and log configuration parameters in a formatted table.
    
    Args:
        **kwargs: Configuration parameters to display
    """
    import sys
    from pathlib import Path
    
    # Create formatted output
    output = []
    output.append('Configurations:')
    output.append('-' * 70)
    output.append(' |%25s | %40s|' % ('keys', 'values'))
    output.append('-' * 70)
    for key, value in kwargs.items():
        output.append(' |%25s | %40s|' % (str(key), str(value)))
    output.append('-' * 70)

    # Print to console
    print('\n'.join(output))

    # Log to file
    log_dir = Path("configs")
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "config.log", "a", encoding="utf-8") as f:
        f.write('\n'.join(output) + '\n\n')


def download_weights(backbone, model_dir="./model_data"):
    """
    Download or load pre-trained model weights.
    
    Args:
        backbone (str): Backbone architecture name
        model_dir (str): Directory to store model weights
        
    Returns:
        torch.Tensor or None: Loaded weights or None if not found
    """
    import os
    import torch

    # Define weight file mappings (to be populated)
    weight_files = {
    }

    weight_path = weight_files.get(backbone)

    if os.path.exists(weight_path):
        print(f"Loading pre-trained weights from {weight_path}")
        return torch.load(weight_path)
    else:
        print(f"Pre-trained weights file {weight_path} not found.")
        return None