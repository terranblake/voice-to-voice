**⚠️ Important**: This tool is for educational and research purposes. Ensure you have permission to use any audio content and respect copyright laws and terms of service.

# Voice Cloning Pipeline

A comprehensive, end-to-end voice cloning pipeline that can extract a speaker's voice from YouTube videos and create a fine-tuned Text-to-Speech (TTS) model capable of generating speech in that voice.

## 🌟 Features

- **YouTube Audio Download**: Automatically download and process audio from YouTube URLs
- **Voice Extraction**: Isolate a specific speaker's voice using AI-powered voice separation
- **Dataset Creation**: Build and upload datasets to Hugging Face Hub
- **Model Fine-tuning**: Fine-tune TTS models using LoRA (Low-Rank Adaptation)
- **Speech Synthesis**: Generate speech samples in the cloned voice
- **Docker Support**: Run everything in containerized environments
- **CLI Interface**: Easy-to-use command-line interface
- **Modular Design**: Use individual components or the complete pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg
- Git
- Hugging Face account and token

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd voice-to-voice
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

3. **Configure your environment**:
   ```bash
   # Edit .env file and add your Hugging Face token
   cp .env.example .env
   # Edit .env and set HF_TOKEN=your_token_here
   ```

4. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

### Basic Usage

**Run the complete pipeline**:
```bash
# Using the legacy script
python main.py "https://www.youtube.com/watch?v=example"

# Using the CLI (recommended)
voice-clone run "https://www.youtube.com/watch?v=example"
```

**Download audio only**:
```bash
voice-clone download "https://www.youtube.com/watch?v=example"
```

**Generate speech from existing model**:
```bash
voice-clone synthesize ./voice_clone/tts_lora --text "Hello, world!"
```

## 📁 Project Structure

```
voice-to-voice/
├── src/
│   └── voice_pipeline/
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration management
│       ├── pipeline.py          # Main pipeline orchestrator
│       ├── utils.py             # Utility functions
│       ├── downloader.py        # YouTube audio downloader
│       ├── extractor.py         # Voice extraction component
│       ├── dataset.py           # Dataset preparation
│       ├── trainer.py           # Model training
│       ├── synthesizer.py       # Speech synthesis
│       └── cli.py              # Command-line interface
├── tests/                       # Test files
├── docs/                        # Documentation
├── main.py                      # Legacy entry point
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── Dockerfile                   # Docker configuration
├── Dockerfile.gpu              # GPU-enabled Docker
├── docker-compose.yml           # Docker Compose setup
├── setup.sh                     # Setup script
└── README.md                    # This file
```

## 🔧 Configuration

The pipeline uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
# Required
HF_TOKEN=your_hugging_face_token_here

# Optional customizations
DEFAULT_OUTPUT_DIR=./voice_clone
DEFAULT_MAX_STEPS=1000
DEFAULT_HF_REPO=username/voice_dataset
DEFAULT_BASE_TTS_MODEL=unsloth/sesame-csm-1b
DEFAULT_REF_START=0
DEFAULT_REF_END=15
```

## 🐳 Docker Usage

### Build and run with Docker Compose:

```bash
# Build the container
docker-compose build

# Run the pipeline
docker-compose run voice-clone "https://www.youtube.com/watch?v=example"

# For GPU support (requires nvidia-docker)
docker-compose --profile gpu run voice-clone-gpu "https://www.youtube.com/watch?v=example"
```

### Direct Docker usage:

```bash
# Build the image
docker build -t voice-clone .

# Run the container
docker run -v $(pwd)/voice_clone:/home/app/voice_clone voice-clone "https://www.youtube.com/watch?v=example"
```

## 🧩 Using Individual Components

```python
from voice_pipeline import PipelineConfig, VoiceCloningPipeline

# Create configuration
config = PipelineConfig(
    yt_url="https://www.youtube.com/watch?v=example",
    hf_repo="myusername/my_voice_dataset",
    max_steps=500
)

# Initialize pipeline
pipeline = VoiceCloningPipeline(config)

# Use individual components
audio_path = pipeline.download_audio()
clips_dir = pipeline.extract_voice(audio_path)
dataset_repo = pipeline.prepare_dataset(clips_dir)
lora_dir = pipeline.train_model(dataset_repo)
demo_path = pipeline.generate_demo(lora_dir, "Custom text to synthesize")
```

## 🔍 Pipeline Steps

1. **Audio Download** (`downloader.py`):
   - Downloads audio from YouTube using `yt-dlp`
   - Converts to WAV format with specified sample rate

2. **Voice Extraction** (`extractor.py`):
   - Creates reference audio slice
   - Uses Spleeter to isolate target speaker
   - Produces segmented audio clips with transcriptions

3. **Dataset Preparation** (`dataset.py`):
   - Formats extracted clips for training
   - Uploads dataset to Hugging Face Hub

4. **Model Training** (`trainer.py`):
   - Fine-tunes base TTS model using LoRA
   - Configurable training parameters

5. **Speech Synthesis** (`synthesizer.py`):
   - Generates demo audio using fine-tuned model
   - Supports custom text input

## 🛠 Development

### Setting up for development:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Running tests:

```bash
pytest tests/ -v --cov=src/voice_pipeline
```

## 📋 Requirements

### System Dependencies:
- FFmpeg
- Git
- Python 3.8+

### Python Dependencies:
- yt-dlp (YouTube downloading)
- ffmpeg-python (Audio processing)
- datasets (HuggingFace datasets)
- transformers (Model inference)
- unsloth (Efficient training)
- spleeter (Voice separation)

## 🚨 Troubleshooting

### Common Issues:

1. **FFmpeg not found**:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu
   sudo apt-get install ffmpeg
   ```

2. **CUDA out of memory**:
   - Reduce batch size in configuration
   - Use CPU-only mode
   - Try the GPU Docker container

3. **Hugging Face authentication**:
   - Ensure HF_TOKEN is set in .env
   - Check token permissions on Hugging Face

4. **Spleeter issues**:
   - Ensure git is installed
   - Check network connectivity
   - Try reinstalling: `pip install spleeter`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- [Spleeter](https://github.com/deezer/spleeter) for voice separation
- [Unsloth](https://github.com/unslothai/unsloth) for efficient model training
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube audio downloading
- Hugging Face ecosystem for model hosting and datasets

## 📧 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**⚠️ Important**: This tool is for educational and research purposes. Ensure you have permission to use any audio content and respect copyright laws and terms of service.
