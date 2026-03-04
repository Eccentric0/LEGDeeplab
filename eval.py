"""
Evaluation script for semantic segmentation models.
This script performs model evaluation and mean Intersection over Union (mIoU) calculation.
"""

import os
from PIL import Image
from tqdm import tqdm
from eval_setting import SegmentationModel
from utils.utils_metrics import compute_mIoU, show_results


if __name__ == "__main__":
    # Evaluation mode configuration
    # 0: Run both prediction and mIoU calculation
    # 1: Run only prediction
    # 2: Run only mIoU calculation
    miou_mode       = 0
    
    # Model configuration
    num_classes     = 3    
    name_classes = ["0", "1", "2"]

    # Dataset paths
    VOCdevkit_path  = 'datasets/VOCdevkit_stripRust_512_743'
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    
    # Output paths
    miou_out_path   = "miou_out/Deeplabv3p_EdgeGuided_test_parameters_Res"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    # Mode 0 or 1: Generate predictions
    if miou_mode == 0 or miou_mode == 1:
        # Create output directories if they don't exist
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Loading segmentation model...")
        net = SegmentationModel()  
        print("Model loaded successfully.")
        
        # Create comparison directory for visualization
        compare_dir = os.path.join(miou_out_path, 'comparison')
        if not os.path.exists(compare_dir):
            os.makedirs(compare_dir)

        print("Generating prediction results...")
        # Process each image in the validation set
        for image_id in tqdm(image_ids):
            # Load and process input image
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = net.get_miou_png(image)

            # Save prediction result
            image.save(os.path.join(pred_dir, image_id + ".png"))

            # Create comparison image (original + ground truth + prediction)
            original_img = Image.open(image_path)
            gt_path = os.path.join(gt_dir, image_id + ".png")
            gt_img = net.visualize_gt(Image.open(gt_path))
            pred_img = net.visualize_gt(image)  

            # Create composite comparison image
            total_width = original_img.width * 3
            max_height = max(original_img.height, gt_img.height, pred_img.height)
            compare_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

            # Arrange images side by side
            compare_img.paste(original_img, (0, 0))
            compare_img.paste(gt_img, (original_img.width, 0))
            compare_img.paste(pred_img, (original_img.width * 2, 0))
            compare_img.save(os.path.join(compare_dir, f"{image_id}_compare.png"))
           
        print("Prediction results generated successfully.")


    # Mode 0 or 2: Calculate mIoU metrics
    if miou_mode == 0 or miou_mode == 2:
        print("Calculating mIoU metrics...")
        # Compute mean Intersection over Union and other metrics
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  
        print("mIoU calculation completed.")
        
        # Display and save results
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)