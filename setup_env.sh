#!/bin/bash
# LEGDeeplab Environment Setup Script
# Sets up a complete development environment for LEGDeeplab

set -e  # Exit on any error

echo "==========================================="
echo "LEGDeeplab Environment Setup"
echo "==========================================="

# Check if running on Windows Subsystem for Linux (WSL)
if grep -q microsoft /proc/version; then
    echo "Detected WSL, proceeding with Linux setup..."
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Function to install conda/miniconda if not present
install_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Installing Miniconda..."
        if [[ "$OS" == "linux" ]]; then
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            bash miniconda.sh -b -p $HOME/miniconda3
        elif [[ "$OS" == "macos" ]]; then
            curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
            bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
        else
            echo "Please install Anaconda/Miniconda manually on Windows"
            exit 1
        fi
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda init bash
    fi
}

# Function to create and activate conda environment
create_environment() {
    ENV_NAME="legdeeplab"
    
    echo "Creating conda environment: $ENV_NAME"
    
    # Create environment with Python 3.8
    conda create -n $ENV_NAME python=3.8 -y
    
    # Activate environment
    conda activate $ENV_NAME
    
    # Install PyTorch with CUDA support (adjust CUDA version as needed)
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install additional packages from conda-forge
    conda install -c conda-forge numpy scipy scikit-image opencv pillow matplotlib seaborn pandas h5py tqdm -y
}

# Function to install Python packages from requirements
install_python_packages() {
    echo "Installing Python packages from requirements.txt..."
    
    # Try to install packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install --no-cache-dir -r requirements.txt
    else
        echo "requirements.txt not found, installing essential packages..."
        pip install --no-cache-dir torch torchvision torchaudio
        pip install --no-cache-dir numpy scipy scikit-image opencv-python
        pip install --no-cache-dir matplotlib seaborn pandas h5py tqdm
        pip install --no-cache-dir tensorboard
    fi
}

# Function to install development tools
install_dev_tools() {
    echo "Installing development tools..."
    pip install --no-cache-dir black flake8 isort mypy pytest pytest-cov pre-commit jupyter notebook
}

# Function to set up project structure
setup_project_structure() {
    echo "Setting up project structure..."
    
    # Create necessary directories
    mkdir -p datasets logs checkpoints outputs img_out
    
    # Initialize git repository if not already done
    if [ ! -d ".git" ]; then
        git init
        echo "Initialized git repository"
    fi
    
    # Setup pre-commit hooks
    pre-commit install
}

# Function to run basic tests
run_basic_tests() {
    echo "Running basic tests..."
    
    # Check if Python can import the main modules
    python -c "import torch; print('PyTorch version:', torch.__version__)"
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    
    # Try to import the main model
    if [ -f "nets/LEGDeeplab.py" ]; then
        python -c "from nets.LEGDeeplab import LEGDeeplab; print('LEGDeeplab imported successfully')"
    else
        echo "LEGDeeplab.py not found"
    fi
}

# Main execution
main() {
    echo "Starting LEGDeeplab environment setup..."
    
    # Install conda if needed
    install_conda
    
    # Source conda (this enables conda command in the script)
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    # Create environment
    create_environment
    
    # Activate the environment
    conda activate legdeeplab
    
    # Install Python packages
    install_python_packages
    
    # Install development tools
    install_dev_tools
    
    # Setup project structure
    setup_project_structure
    
    # Run basic tests
    run_basic_tests
    
    echo ""
    echo "==========================================="
    echo "LEGDeeplab Environment Setup Complete!"
    echo "==========================================="
    echo ""
    echo "To activate the environment in the future, run:"
    echo "  conda activate legdeeplab"
    echo ""
    echo "To deactivate the environment, run:"
    echo "  conda deactivate"
    echo ""
    echo "The following directories have been created:"
    echo "  datasets/: For storing datasets"
    echo "  logs/: For storing training logs"
    echo "  checkpoints/: For saving model checkpoints"
    echo "  outputs/: For storing outputs"
    echo "  img_out/: For storing image predictions"
    echo ""
    echo "Ready to use LEGDeeplab! You can now run:"
    echo "  python train.py    # To train the model"
    echo "  python eval.py     # To evaluate the model"
    echo "  python benchmark.py # To benchmark the model"
    echo ""
}

# Run main function
main "$@"