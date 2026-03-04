"""
LEGDeeplab: Advanced Evaluation Script for Semantic Segmentation
==============================================================

This comprehensive evaluation script provides multiple evaluation modes for semantic segmentation models:
- mIoU calculation with detailed metrics
- Prediction visualization
- Performance benchmarking
- Model comparison
- Advanced metric computation

Authors: LEGDeeplab Development Team
License: MIT
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nets.LEGDeeplab import LEGDeeplab
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics import compute_mIoU, evalute


def parse_arguments():
    """Parse command line arguments for flexible evaluation configuration."""
    parser = argparse.ArgumentParser(description='Evaluate LEGDeeplab for Semantic Segmentation')
    
    # Evaluation mode
    parser.add_argument('--miou_mode', type=int, default=0, choices=[0, 1, 2], 
                        help='Evaluation mode: 0-compute mIoU, 1-generate predictions, 2-calculate metrics')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of segmentation classes')
    parser.add_argument('--model_path', type=str, default='', help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='./datasets/VOCdevkit', help='Path to dataset')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[512, 512], help='Input image dimensions')
    
    # Dataset configuration
    parser.add_argument('--gt_dir', type=str, default='VOC2007/SegmentationClass', help='Ground truth directory')
    parser.add_argument('--pred_dir', type=str, default='img_out', help='Prediction output directory')
    parser.add_argument('--image_dir', type=str, default='VOC2007/JPEGImages', help='Image input directory')
    parser.add_argument('--txt_path', type=str, default='VOC2007/ImageSets/Segmentation/val.txt', help='Text file with image IDs')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='LEGDeeplab', help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone network')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize predictions')
    parser.add_argument('--count', action='store_true', default=False, help='Count class distribution')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save evaluation results')
    
    # Advanced metrics
    parser.add_argument('--compute_advanced_metrics', action='store_true', default=True, help='Compute advanced metrics')
    parser.add_argument('--save_confusion_matrix', action='store_true', default=False, help='Save confusion matrix')
    
    # Hardware configuration
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()


class SemanticSegmentationEvaluator:
    """Advanced evaluator for semantic segmentation models."""
    
    def __init__(self, model, input_shape, num_classes, cuda=True):
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cuda = cuda
        
        # Initialize metrics storage
        self.predictions = []
        self.ground_truths = []
        self.class_histograms = np.zeros(num_classes)
        
    def predict_image(self, image):
        """Predict segmentation mask for a single image."""
        # Preprocess image
        image_data = cvtColor(image)
        image_data = np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        
        with torch.no_grad():
            photos = torch.from_numpy(image_data)
            if self.cuda:
                photos = photos.cuda()
                
            # Get prediction
            preds = self.model(photos)[0]
            pred = F.softmax(preds.permute(1, 2, 0), dim=-1).cpu().numpy()
            pred = np.argmax(pred, axis=-1)
        
        return pred
    
    def compute_advanced_metrics(self, y_true, y_pred):
        """Compute advanced segmentation metrics."""
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(self.num_classes))
        
        # Compute metrics per class
        metrics = {}
        
        # Overall accuracy
        overall_acc = np.diag(cm).sum() / cm.sum()
        
        # Per-class metrics
        class_precision = np.zeros(self.num_classes)
        class_recall = np.zeros(self.num_classes)
        class_f1 = np.zeros(self.num_classes)
        class_iou = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            class_precision[i] = precision
            class_recall[i] = recall
            class_f1[i] = f1
            class_iou[i] = iou
        
        # Mean metrics
        mean_precision = np.mean(class_precision)
        mean_recall = np.mean(class_recall)
        mean_f1 = np.mean(class_f1)
        mean_iou = np.mean(class_iou)
        
        # Frequency Weighted IoU
        freq = cm.sum(axis=1) / cm.sum()
        fw_iou = (freq[freq > 0] * class_iou[freq > 0]).sum()
        
        metrics = {
            'overall_accuracy': overall_acc,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou,
            'fw_iou': fw_iou,
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'class_iou': class_iou.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def evaluate_model(self, image_ids, image_dir, gt_dir, pred_dir=None, visualize=False):
        """Comprehensive model evaluation."""
        print("Starting evaluation...")
        
        # Create prediction directory if needed
        if pred_dir and not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        all_metrics = []
        progress_bar = tqdm(total=len(image_ids), desc='Evaluating', ncols=100)
        
        for i, img_id in enumerate(image_ids):
            img_id = img_id.strip()
            
            # Load image and ground truth
            image_path = os.path.join(image_dir, img_id + ".jpg")
            gt_path = os.path.join(gt_dir, img_id + ".png")
            
            image = Image.open(image_path)
            gt_mask = np.array(Image.open(gt_path))
            gt_mask = cv2.resize(gt_mask, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Predict
            pred_mask = self.predict_image(image)
            
            # Store for later analysis
            self.ground_truths.append(gt_mask)
            self.predictions.append(pred_mask)
            
            # Count class distribution
            unique, counts = np.unique(pred_mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < self.num_classes:
                    self.class_histograms[u] += c
            
            # Save prediction if needed
            if pred_dir:
                pred_save_path = os.path.join(pred_dir, img_id + ".png")
                pred_img = Image.fromarray(pred_mask.astype('uint8'))
                pred_img.save(pred_save_path)
            
            # Visualize if requested
            if visualize:
                self.visualize_prediction(image, gt_mask, pred_mask, img_id)
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Compute final metrics
        if self.compute_advanced_metrics:
            all_gt = np.concatenate(self.ground_truths)
            all_pred = np.concatenate(self.predictions)
            final_metrics = self.compute_advanced_metrics(all_gt, all_pred)
        else:
            # Use traditional mIoU computation
            miou_out_path = pred_dir if pred_dir else './temp_predictions'
            if not os.path.exists(miou_out_path):
                os.makedirs(miou_out_path)
            
            # Save all predictions temporarily for mIoU computation
            for i, (img_id, pred_mask) in enumerate(zip(image_ids, self.predictions)):
                pred_img = Image.fromarray(pred_mask.astype('uint8'))
                pred_img.save(os.path.join(miou_out_path, img_id.strip() + ".png"))
            
            # Compute mIoU using traditional method
            _, IoUs, _, _ = compute_mIoU(
                gt_dir, miou_out_path, len(image_ids), self.num_classes, None
            )
            final_metrics = {'mean_iou': np.nanmean(IoUs)}
        
        return final_metrics
    
    def visualize_prediction(self, image, gt_mask, pred_mask, img_id):
        """Visualize prediction vs ground truth."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(gt_mask, cmap='tab20')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_mask, cmap='tab20')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        vis_path = f'visualization_{img_id}.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_confusion_matrix(self, metrics, save_path):
        """Save confusion matrix as heatmap."""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(self.num_classes), 
                    yticklabels=range(self.num_classes))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Load model
    if args.model_name == "LEGDeeplab":
        model = LEGDeeplab(
            in_channels=3, 
            num_classes=args.num_classes, 
            backbone=args.backbone
        )
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.eval()
    
    if args.cuda:
        model = model.cuda()
    
    print(f"Model loaded from: {args.model_path}")
    
    # Initialize evaluator
    evaluator = SemanticSegmentationEvaluator(
        model=model,
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        cuda=args.cuda
    )
    
    # Read image IDs
    with open(os.path.join(args.dataset_path, args.txt_path), 'r') as f:
        image_ids = f.readlines()
    
    # Prepare paths
    image_dir = os.path.join(args.dataset_path, args.image_dir)
    gt_dir = os.path.join(args.dataset_path, args.gt_dir)
    pred_dir = os.path.join(args.dataset_path, args.pred_dir) if args.miou_mode in [0, 1] else None
    
    # Perform evaluation based on mode
    if args.miou_mode == 0:  # Calculate mIoU and generate predictions
        print("Calculating mIoU and generating predictions...")
        metrics = evaluator.evaluate_model(
            image_ids=image_ids,
            image_dir=image_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            visualize=args.visualize
        )
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Mean F1-Score: {metrics['mean_f1']:.4f}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"Frequency Weighted IoU: {metrics['fw_iou']:.4f}")
        
        # Save results
        if args.save_results:
            results_path = os.path.join(args.dataset_path, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Results saved to: {results_path}")
        
        # Save confusion matrix if requested
        if args.save_confusion_matrix:
            cm_path = os.path.join(args.dataset_path, 'confusion_matrix.png')
            evaluator.save_confusion_matrix(metrics, cm_path)
            print(f"Confusion matrix saved to: {cm_path}")
    
    elif args.miou_mode == 1:  # Generate predictions only
        print("Generating predictions...")
        metrics = evaluator.evaluate_model(
            image_ids=image_ids,
            image_dir=image_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            visualize=args.visualize
        )
        print("Predictions generated successfully!")
    
    elif args.miou_mode == 2:  # Calculate metrics only
        print("Calculating metrics...")
        # This would typically load pre-generated predictions
        # For now, we'll run the full evaluation to get metrics
        metrics = evaluator.evaluate_model(
            image_ids=image_ids,
            image_dir=image_dir,
            gt_dir=gt_dir,
            pred_dir=None,
            visualize=False
        )
        
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Mean F1-Score: {metrics['mean_f1']:.4f}")
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()