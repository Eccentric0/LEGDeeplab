
# Contribution Guide for LEGDeeplab

## Welcome Contributors!

Thank you for your interest in contributing to the LEGDeeplab project! This repository contains the official implementation of the lightweight edge-guided DeepLabv3+ model for semantic segmentation, optimized for real-time deployment in agricultural applications such as maize leaf rust detection.

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Basic understanding of PyTorch and semantic segmentation.

### Setting Up Your Environment
```bash
# Clone the repository
git clone https://github.com/zhou/SegmentationNet.git
cd SegmentationNet

# Install dependencies
pip install -r requirements.txt
```

## Development Workflow
We welcome feature contributions and bug fixes! For larger changes, please open an issue first.

### Branching Strategy
- `main`: Stable releases
- `develop`: Integration branch for new features

### Coding Standards
- Follow PEP8 for Python code style.
- Ensure all public functions and methods have Google-style docstrings.

### Testing Requirements
All contributions must include unit tests for model components.
