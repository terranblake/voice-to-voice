#!/bin/bash
set -e

echo "🚀 Setting up Voice Cloning Pipeline..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📥 Installing package dependencies..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install -e ".[dev]"

# Install system dependencies reminder
echo ""
echo "⚠️  IMPORTANT: Please ensure the following system dependencies are installed:"
echo "   - ffmpeg (for audio processing)"
echo "   - git (for Voice_Extractor installation)"
echo ""
echo "On macOS: brew install ffmpeg"
echo "On Ubuntu: sudo apt-get install ffmpeg"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file and add your Hugging Face token!"
fi

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

echo ""
echo "✅ Setup complete! Next steps:"
echo "   1. Edit .env file and add your HF_TOKEN"
echo "   2. Activate virtual environment: source venv/bin/activate"
echo "   3. Run the pipeline: python main.py <youtube_url>"
echo "   4. Or use the CLI: voice-clone run <youtube_url>"
echo ""
