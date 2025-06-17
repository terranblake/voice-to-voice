# Development Documentation

## Project Architecture

The Voice Cloning Pipeline is designed with modularity and maintainability in mind. Each component has a specific responsibility and can be used independently or as part of the full pipeline.

### Core Components

#### 1. Configuration (`config.py`)
- Centralizes all configuration management
- Uses environment variables with sensible defaults
- Validates configuration on initialization
- Supports both programmatic and file-based configuration

#### 2. Pipeline Orchestrator (`pipeline.py`)
- Main class that coordinates all components
- Provides both full pipeline execution and individual component access
- Handles error propagation and logging
- Maintains state between pipeline steps

#### 3. Individual Components
Each component follows the same pattern:
- Takes configuration and logger in constructor
- Has a main public method for its core functionality
- Handles its own error checking and recovery
- Returns paths or identifiers for next pipeline step

### Component Details

#### YouTube Downloader (`downloader.py`)
```python
class YouTubeDownloader:
    def download(self) -> Path:
        # Downloads audio, converts to WAV
        # Returns path to processed audio file
```

#### Voice Extractor (`extractor.py`)
```python
class VoiceExtractor:
    def extract_voice(self, source_wav: Path) -> Path:
        # Creates reference slice
        # Runs voice separation
        # Returns path to dataset directory
```

#### Dataset Preparer (`dataset.py`)
```python
class DatasetPreparer:
    def prepare_and_upload(self, dataset_dir: Path) -> str:
        # Collects audio/text pairs
        # Creates HuggingFace dataset
        # Uploads to Hub
        # Returns repository name
```

#### TTS Trainer (`trainer.py`)
```python
class TTSTrainer:
    def train(self, dataset_repo: str) -> Path:
        # Loads base model and dataset
        # Configures LoRA training
        # Executes training loop
        # Returns path to saved adapter
```

#### Speech Synthesizer (`synthesizer.py`)
```python
class SpeechSynthesizer:
    def generate_demo(self, lora_dir: Path, base_model: str, text: str) -> Path:
        # Loads model with adapter
        # Generates audio from text
        # Returns path to audio file
```

## Development Workflow

### Setting Up Development Environment

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd voice-to-voice
   ./setup.sh
   source venv/bin/activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

### Code Style

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

Run all checks:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/voice_pipeline --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

### Adding New Components

1. Create new module in `src/voice_pipeline/`
2. Follow the established pattern:
   ```python
   class NewComponent:
       def __init__(self, config: PipelineConfig, logger: logging.Logger):
           self.config = config
           self.logger = logger
       
       def main_method(self, input_data) -> output_type:
           # Implementation
           pass
   ```
3. Add to pipeline in `pipeline.py`
4. Write tests in `tests/`
5. Update documentation

### Debugging

Enable verbose logging:
```bash
# CLI
voice-clone run --verbose <url>

# Environment variable
export LOG_LEVEL=DEBUG
```

Use individual components for testing:
```python
# Test just the downloader
config = PipelineConfig(yt_url="...")
pipeline = VoiceCloningPipeline(config)
audio_path = pipeline.download_audio()
```

## Performance Considerations

### Memory Usage
- Training can be memory-intensive
- Use gradient accumulation to reduce batch size
- Consider mixed precision training (fp16/bf16)

### Storage
- Downloaded audio files can be large
- Model checkpoints require significant space
- Clean up intermediate files when possible

### GPU Utilization
- Training benefits significantly from GPU
- Inference can run on CPU but slower
- Use appropriate Docker container for your setup

## Configuration Management

### Environment Variables
All configuration uses environment variables with the `DEFAULT_` prefix:

```bash
# Core settings
DEFAULT_OUTPUT_DIR=./voice_clone
DEFAULT_HF_REPO=username/dataset
DEFAULT_BASE_TTS_MODEL=unsloth/sesame-csm-1b

# Training settings
DEFAULT_MAX_STEPS=1000
DEFAULT_BATCH_SIZE=2
DEFAULT_GRADIENT_ACCUMULATION=8
DEFAULT_LEARNING_RATE=2e-4

# Audio settings
AUDIO_SAMPLE_RATE=44100
TTS_SAMPLE_RATE=24000
```

### Programmatic Configuration
```python
from voice_pipeline import PipelineConfig

config = PipelineConfig(
    yt_url="https://youtube.com/watch?v=...",
    out_dir=Path("custom_output"),
    hf_repo="myuser/mydataset",
    max_steps=500,
    ref_seconds=(10, 25)  # Use 10-25 seconds as reference
)
```

## Error Handling

The pipeline uses a consistent error handling strategy:

1. **Validation errors**: Raise `ValueError` with descriptive message
2. **System errors**: Log error and call `sys.exit(1)` 
3. **Recoverable errors**: Log warning and continue with fallback
4. **External tool errors**: Capture subprocess errors and provide helpful messages

Example:
```python
try:
    result = external_tool_call()
except subprocess.CalledProcessError as e:
    self.logger.error(f"Tool failed: {e}")
    self.logger.error("Please check that tool is properly installed")
    sys.exit(1)
```

## Extending the Pipeline

### Adding New Audio Sources
Create a new downloader component:
```python
class SpotifyDownloader:
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        # Implementation
    
    def download(self) -> Path:
        # Download from Spotify
        # Return path to audio file
```

### Adding New TTS Models
Extend the trainer or create model-specific trainers:
```python
class CustomTTSTrainer(TTSTrainer):
    def train(self, dataset_repo: str) -> Path:
        # Custom training logic for specific model
        pass
```

### Adding New Voice Extractors
Create alternative extraction methods:
```python
class AlternativeVoiceExtractor:
    def extract_voice(self, source_wav: Path) -> Path:
        # Different voice separation algorithm
        pass
```

## Deployment

### Docker Best Practices
- Use multi-stage builds for smaller images
- Set proper user permissions (non-root)
- Mount volumes for persistent data
- Use .dockerignore to exclude unnecessary files

### Production Considerations
- Set up proper logging aggregation
- Monitor resource usage
- Implement health checks
- Consider horizontal scaling for batch processing

### Security
- Never commit tokens or credentials
- Use environment variables for secrets
- Validate all user inputs
- Keep dependencies updated
