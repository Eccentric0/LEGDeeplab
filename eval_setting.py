"""
Segmentation model configuration and inference settings.
This module defines the SegmentationModel class for loading and using segmentation models.
"""

import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.LEGDeeplab import LEGDeeplab as Deeplabv3p_EdgeGuided_test_parameters_Res_边缘算子_important

from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class SegmentationModel(object):
    """
    A wrapper class for semantic segmentation models.
    Handles model loading, preprocessing, inference, and visualization.
    """
    
    # Default configuration parameters
    _defaults = {
        "model_path": '',           # Path to pre-trained model weights
        "num_classes": 3,           # Number of segmentation classes
        "model_type": "LEGDeeplab", # Type of segmentation model
        "backbone": "",             # Backbone network architecture
        "input_shape": [512, 512],  # Input image dimensions [height, width]
        "mix_type": 0,              # Data augmentation type
        "cuda": True,               # Whether to use GPU acceleration
    }

    def __init__(self, **kwargs):
        """
        Initialize the segmentation model with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to override defaults
        """
        # Set default parameters
        self.__dict__.update(self._defaults)
        
        # Override with provided parameters
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        # Generate color palette for visualization
        if self.num_classes <= 21:
            # Use predefined colors for small number of classes
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                           (128, 64, 12)]
        else:
            # Generate colors using HSV color space for larger number of classes
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
            
        # Initialize the model
        self.generate()

        # Display configuration
        show_config(**self._defaults)

    def generate(self, onnx=False):
        """
        Load and initialize the segmentation model.
        
        Args:
            onnx (bool): Whether to export to ONNX format
        """
        # Initialize model based on type
        if self.model_type == "Deeplabv3p_EdgeGuided_test_parameters_Res_边缘算子_important":
            self.net = Deeplabv3p_EdgeGuided_test_parameters_Res_边缘算子_important(num_classes=self.num_classes, in_channels=3).train()
        else:
            raise ValueError(f"Unsupported model type - `{self.model_type}`, Use unet, deeplab, pspnet, LinkNet, SegFormer, PDSNet")

        # Set device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained weights if available
        self.net.load_state_dict(
            torch.load(self.model_path,
            map_location=device),
            strict=False  
        )
        loaded_keys = set(torch.load(self.model_path).keys())
        current_keys = set(self.net.state_dict().keys())
        print("Missing keys:", loaded_keys - current_keys)
        print("Unexpected keys:", current_keys - loaded_keys)

        # Set model to evaluation mode
        self.net = self.net.eval()
        print(f'{self.model_path} model, and classes loaded.')
        
        # Enable GPU acceleration if available and not exporting to ONNX
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def visualize_gt(self, gt_image):
        """
        Convert ground truth segmentation mask to color image.
        
        Args:
            gt_image (PIL.Image): Grayscale ground truth image
            
        Returns:
            PIL.Image: Colorized segmentation mask
        """
        gt_array = np.array(gt_image)
        color_seg = np.zeros((gt_array.shape[0], gt_array.shape[1], 3), dtype=np.uint8)
        
        # Apply color mapping for each class
        for cls in range(self.num_classes):
            color_seg[gt_array == cls] = self.colors[cls]
        return Image.fromarray(color_seg)

    def detect_image(self, image, count=False, name_classes=None):
        """
        Perform semantic segmentation inference on an input image.
        
        Args:
            image (PIL.Image): Input image to segment
            count (bool): Whether to count pixel distribution per class
            name_classes (list): List of class names for counting
            
        Returns:
            PIL.Image: Segmented image
        """
        # Preprocess input image
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        
        # Resize and normalize image
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # Perform inference
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            
            # Convert logits to probabilities and resize to original dimensions
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)  # Get class predictions

        # Count pixel distribution per class if requested
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        image = Image.fromarray(np.uint8(pr), mode='L')

        # Apply different visualization modes
        if self.mix_type == 0:
            # Blend original image with segmentation mask
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)  # 70% original + 30% segmentation
        elif self.mix_type == 1:
            # Pure segmentation mask
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            # Highlight foreground on original image
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image


    def get_FPS(self, image, test_interval):
        """
        Calculate inference speed (frames per second).
        
        Args:
            image (PIL.Image): Test image for timing
            test_interval (int): Number of iterations for timing
            
        Returns:
            float: Average inference time per frame (seconds)
        """
        # Preprocess input image
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # Warm-up inference
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        # Time multiple inferences for accurate measurement
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            simplify (bool): Whether to simplify the ONNX model
            model_path (str): Output path for ONNX model
        """
        import onnx
        
        # Initialize model for ONNX export
        self.generate(onnx=True)

        # Create dummy input for tracing
        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export model to ONNX format
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Validate ONNX model
        model_onnx = onnx.load(model_path)  
        onnx.checker.check_model(model_onnx)  

        # Simplify model if requested
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        """
        Generate segmentation mask for mIoU calculation.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Grayscale segmentation mask
        """
        # Preprocess input image
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # Perform inference
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]  
            if len(pr.shape) == 4:      # Handle batch dimension
                pr = pr.squeeze(0)      # Remove batch dimension
            
            # Convert to probabilities and resize
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)  # Get class predictions

        # Return grayscale segmentation mask
        image = Image.fromarray(np.uint8(pr))
        return image