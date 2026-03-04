
# Changelog

## [Unreleased]

### Added
- Edge-Guided attention mechanisms for better boundary segmentation.
- Enhanced lightweight design with ResNet18 backbone.
- New edge feature aggregation (EFA) module for improved boundary detection.

### Changed
- Updated model architecture with additional lightweight modules (LSK, ScConv, AGCA).
- Improved performance on maize leaf rust detection task.

### Fixed
- Reduced model complexity with only 1.56M parameters.
- Optimized for real-time deployment with computational cost of 14.20 GFLOPs.

## [1.0.0] - 2024-01-15

### Added
- LEGDeeplab model architecture with LSK, ScConv, AGCA, and EFA modules.
- Model tested on a self-built maize leaf rust dataset (743 images) achieving 88.67% mIoU.

### Changed
- Refined edge-guided segmentation with Sobel operator for improved accuracy.

### Fixed
- Reduced memory usage and enhanced generalization across different plant disease datasets.
