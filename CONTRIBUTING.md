# Contribution Guide for LEGDeeplab

## Welcome Contributors!

Thank you for your interest in contributing to the LEGDeeplab project! This document outlines the process and standards for contributing to this semantic segmentation framework.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Understanding of PyTorch and semantic segmentation concepts
- Familiarity with the LEGDeeplab architecture

### Setting Up Your Environment

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/SegmentationNet.git
cd SegmentationNet

# 3. Create virtual environment
python -m venv legdeeplab_env
source legdeeplab_env/bin/activate  # On Windows: legdeeplab_env\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install in development mode
pip install -e .
```

## Development Workflow

### Branch Strategy

- `main`: Stable releases
- `develop`: Integration branch for new features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-awesome-feature
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) guidelines
- Use [Google Style](https://google.github.io/styleguide/pyguide.html) docstrings
- Maximum line length: 88 characters
- Use type hints for all public functions

### Code Structure

```python
# Example of proper code structure
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

class MyAwesomeModule(nn.Module):
    """Brief description of the module.
    
    Detailed description of the module, including:
    - Purpose and functionality
    - Mathematical background if applicable
    - Key design decisions
    
    Args:
        arg1: Description of first argument
        arg2: Description of second argument
    
    Attributes:
        attr1: Description of first attribute
        attr2: Description of second attribute
    """
    
    def __init__(self, 
                 arg1: int, 
                 arg2: Optional[str] = None,
                 **kwargs) -> None:
        """Initialize the module.
        
        Args:
            arg1: Description of first argument
            arg2: Description of second argument
            **kwargs: Additional keyword arguments
        """
        super(MyAwesomeModule, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        
        # Implementation here
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        # Implementation here
        return output
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `LargeSelectiveKernel`)
- **Functions/Methods**: `snake_case` (e.g., `compute_attention`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_INPUT_SIZE`)
- **Private members**: Prefix with underscore (e.g., `_hidden_features`)

## Testing Requirements

### Unit Tests

All contributions must include comprehensive unit tests:

```python
import unittest
import torch
from torch.testing import assert_close

from nets.my_module import MyAwesomeModule

class TestMyAwesomeModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.module = MyAwesomeModule(arg1=64)
        self.input_tensor = torch.randn(2, 3, 224, 224)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        output = self.module(self.input_tensor)
        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        # Additional assertions...
    
    def test_different_input_sizes(self):
        """Test with various input sizes."""
        for height, width in [(64, 64), (128, 256), (512, 512)]:
            input_tensor = torch.randn(1, 3, height, width)
            output = self.module(input_tensor)
            # Assertions for expected behavior
    
    def test_dtype_consistency(self):
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float16]:
            input_tensor = self.input_tensor.to(dtype=dtype)
            output = self.module(input_tensor)
            self.assertEqual(output.dtype, dtype)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_my_module.py

# Run with coverage
python -m pytest tests/ --cov=nets --cov=utils --cov-report=html
```

## Documentation Standards

### Docstrings

All public functions, classes, and modules must have comprehensive docstrings following the Google style format:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """Brief description of the function.
    
    Detailed explanation of what the function does, including:
    - Algorithm description
    - Mathematical formulation if applicable
    - Important considerations
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter, with default value
        
    Returns:
        Description of the return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
        
    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)
        True
    """
    # Implementation
    pass
```

### README Updates

When adding new features:
- Update the main README with usage examples
- Add performance benchmarks if applicable
- Include any new configuration options
- Update the architecture diagram if necessary

## Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   python -m pytest tests/
   # Ensure all tests pass
   ```

2. **Format your code**:
   ```bash
   black .
   isort .
   flake8 .
   ```

3. **Update documentation**:
   - Add docstrings to new functions/classes
   - Update README if necessary
   - Add usage examples

4. **Check performance**:
   - Ensure no performance regression
   - Add benchmark results if improving performance

### Creating the Pull Request

1. **Push your branch**:
   ```bash
   git add .
   git commit -m "feat: Add awesome new feature"
   git push origin feature/my-awesome-feature
   ```

2. **Submit PR**:
   - Go to the repository on GitHub
   - Click "Compare & pull request"
   - Fill out the PR template:
     - **Title**: Clear, descriptive title
     - **Description**: What was changed and why
     - **Related Issues**: Link any related issues
     - **Checklist**: Confirm all requirements met

### PR Review Process

1. **Initial Check**: Maintainers verify basic requirements
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Automated tests run on CI
4. **Approval**: At least one maintainer approves
5. **Merge**: Maintainers merge the PR

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, PyTorch version, OS
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable
- **System information**: GPU type, memory, etc.

### Feature Requests

When requesting features, please include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How would it work?
- **Use cases**: When would this be useful?
- **Alternatives considered**: Other approaches evaluated

## Recognition

Contributors will be recognized in:
- The README's contributors section
- Release notes
- Academic citations when applicable

## Questions?

If you have questions about contributing:
- Check the existing issues and discussions
- Open a discussion thread
- Contact the maintainers directly

---

Thank you for contributing to LEGDeeplab! Your efforts help advance semantic segmentation research and development.