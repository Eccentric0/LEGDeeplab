"""
LEGDeeplab - Setup Configuration
===============================

Setup script for installing the LEGDeeplab semantic segmentation framework.
This makes the package installable via pip and defines package metadata.

Authors: LEGDeeplab Development Team
Version: 1.0.0
License: MIT
"""

import os
from setuptools import setup, find_packages


# Read the contents of README file
def read_long_description():
    """Read the long description from README.md."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


# Get version from a version file or define it here
VERSION = "1.0.0"
DESCRIPTION = "LEGDeeplab: Lightweight Edge-Guided DeepLabv3+ for Advanced Semantic Segmentation"


setup(
    name="legdeeplab",
    version=VERSION,
    author="LEGDeeplab Development Team",
    author_email="contact@legdeeplab.org",
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "examples*", "benchmarks*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "semantic segmentation",
        "computer vision", 
        "deep learning",
        "pytorch",
        "attention mechanisms",
        "edge detection",
        "real-time inference"
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
            "pre-commit>=2.17",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
            "myst-parser>=0.17",
        ],
        "deploy": [
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legdeeplab-train = scripts.train:main",
            "legdeeplab-eval = scripts.eval:main",
            "legdeeplab-export = scripts.export:main",
            "legdeeplab-benchmark = scripts.benchmark:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/zhou/SegmentationNet/issues",
        "Source": "https://github.com/zhou/SegmentationNet",
        "Documentation": "https://legdeeplab.readthedocs.io/",
        "Homepage": "https://github.com/zhou/SegmentationNet",
    },
    license="MIT",
    license_files=("LICENSE",),
    include_package_data=True,
    zip_safe=False,
)