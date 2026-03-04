# LEGDeeplab Makefile
# Common operations for development and deployment

.PHONY: help install test train evaluate export clean docs format lint

# Default target
help:
	@echo "LEGDeeplab - Makefile for common operations"
	@echo ""
	@echo "Usage:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make train        - Start training"
	@echo "  make evaluate     - Run evaluation"
	@echo "  make export       - Export model to various formats"
	@echo "  make clean        - Clean temporary files"
	@echo "  make docs         - Build documentation"
	@echo "  make format       - Format code"
	@echo "  make lint         - Lint code"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	python -m pytest tests/ -v

# Run training
train:
	python train.py

# Run evaluation
evaluate:
	python eval.py

# Export model
export:
	python export.py --format onnx
	python export.py --format tensorrt

# Clean temporary files
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf logs/*
	rm -rf checkpoints/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Build documentation
docs:
	pdoc --html --output-dir docs/ nets/ utils/

# Format code
format:
	black .
	isort .

# Lint code
lint:
	flake8 .
	mypy .

# Docker operations
docker-build:
	docker build -t legdeeplab:latest .

docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-p 8888:8888 \
		legdeeplab:latest

# Advanced training configurations
train-large:
	python train.py --batch_size 16 --input_shape 640 640

train-small:
	python train.py --batch_size 4 --input_shape 320 320

# Advanced evaluation configurations
eval-fast:
	python eval.py --miou_mode 1 --num_classes 21

eval-comprehensive:
	python eval.py --miou_mode 0 --num_classes 21 --visualize True

# Model download and setup
setup-datasets:
	@echo "Setting up datasets directory structure..."
	mkdir -p datasets/VOCdevkit
	mkdir -p logs
	mkdir -p checkpoints
	mkdir -p outputs

# Performance profiling
profile:
	python -m cProfile -o profile.stats train.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Environment setup for different platforms
setup-linux:
	@echo "Setting up for Linux platform..."
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

setup-windows:
	@echo "Setting up for Windows platform..."
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

setup-macos:
	@echo "Setting up for macOS platform..."
	pip install torch torchvision torchaudio

# Benchmarking
benchmark:
	@echo "Running benchmark tests..."
	python benchmark.py --models LEGDeeplab DeepLabv3+ PSPNet --datasets voc2012 cityscapes

# Advanced operations
debug-mode:
	python -O train.py --debug True

production-mode:
	python train.py --fp16 True --deterministic False

# Help target
help-all:
	@echo "All available targets:"
	@$(MAKE) -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}'

# Default to help
.DEFAULT_GOAL := help