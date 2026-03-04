
# LEGDeeplab: Lightweight Edge-Guided DeepLabv3+ for Semantic Segmentation

**LEGDeeplab** is a cutting-edge semantic segmentation model based on an improved DeepLabv3+ architecture. It leverages edge-guided attention mechanisms and lightweight modules to offer high accuracy and efficiency, making it ideal for real-time applications such as detecting maize leaf rust in agricultural fields.

## Key Features
- **Edge-Guided Attention**: Helps enhance boundary segmentation in noisy field images.
- **Lightweight Architecture**: Reduced model size (1.56M parameters) and computational cost (14.20 GFLOPs).
- **Advanced Attention Modules**: Large Selective Kernel, Sparse Convolution, and Adaptive Graph Convolution Attention.

## Installation
```bash
git clone https://github.com/Eccentric0/LEGDeeplab
cd SegmentationNet
pip install -r requirements.txt
```

## Quick Start
```bash
# Train the model
python train.py --model_name LEGDeeplab  --num_classes 3

# Evaluate the model
python eval.py --miou_mode 0 --num_classes 3
```

## Results
LEGDeeplab achieves superior performance on the maize leaf rust dataset with 88.67% mIoU.

## Citation
If you use LEGDeeplab in your research, please cite the following paper:
"LEGDeeplab: A Lightweight Edge-Guided Model for Semantic Segmentation of Maize Leaf Rust", Zhenbang Zhou, et al.
