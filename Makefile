# Voice Cloning Pipeline Makefile

.PHONY: help install install-dev setup test lint format clean docker docker-gpu run

# Default target
help:
	@echo "Voice Cloning Pipeline - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup        - Complete project setup (creates venv, installs deps)"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  format       - Format code (black, isort)"
	@echo "  clean        - Clean up generated files"
	@echo ""
	@echo "Docker:"
	@echo "  docker       - Build Docker image"
	@echo "  docker-gpu   - Build GPU Docker image"
	@echo ""
	@echo "Usage:"
	@echo "  run URL      - Run pipeline with YouTube URL"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make run URL='https://youtube.com/watch?v=example'"

# Setup complete environment
setup:
	@echo "🚀 Setting up Voice Cloning Pipeline..."
	./setup.sh

# Install package
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=src/voice_pipeline --cov-report=term-missing

# Run linting
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf voice_clone/

# Build Docker image
docker:
	docker build -t voice-clone .

# Build GPU Docker image
docker-gpu:
	docker build -f Dockerfile.gpu -t voice-clone-gpu .

# Run pipeline with URL parameter
run:
	@if [ -z "$(URL)" ]; then \
		echo "❌ Error: URL parameter is required"; \
		echo "Usage: make run URL='https://youtube.com/watch?v=example'"; \
		exit 1; \
	fi
	@echo "🎵 Running voice cloning pipeline..."
	python main.py "$(URL)"

# Development server (if needed for future web interface)
dev-server:
	@echo "Development server not implemented yet"

# Check all (comprehensive check before commit)
check-all: format lint test
	@echo "✅ All checks passed!"

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files
