# API Reference

## Main Classes

### PipelineConfig

Configuration class for the voice cloning pipeline.

```python
@dataclass
class PipelineConfig:
    yt_url: str                                    # Required: YouTube URL
    ref_seconds: Tuple[int, int] = (0, 15)        # Reference audio slice
    out_dir: Path = Path("voice_clone")           # Output directory
    hf_repo: str = "username/female_voice_ds"     # HuggingFace repository
    base_tts: str = "unsloth/sesame-csm-1b"      # Base TTS model
    max_steps: int = 1000                         # Training steps
    hf_token: Optional[str] = None                # HuggingFace token
    # ... additional configuration options
```

**Methods:**
- `__post_init__()`: Validates configuration after initialization

### VoiceCloningPipeline

Main pipeline orchestrator.

```python
class VoiceCloningPipeline:
    def __init__(self, config: PipelineConfig, logging_config: Optional[LoggingConfig] = None)
```

**Methods:**

#### `run_full_pipeline(demo_text: str = "...") -> Path`
Executes the complete voice cloning pipeline.

**Parameters:**
- `demo_text`: Text to synthesize for the demo

**Returns:**
- Path to the generated demo audio file

#### `download_audio() -> Path`
Downloads audio from YouTube URL.

**Returns:**
- Path to the downloaded and converted WAV file

#### `extract_voice(source_wav: Path) -> Path`
Extracts target voice from the source audio.

**Parameters:**
- `source_wav`: Path to the source WAV file

**Returns:**
- Path to the directory containing extracted voice clips

#### `prepare_dataset(clips_dir: Path) -> str`
Prepares dataset and uploads to Hugging Face Hub.

**Parameters:**
- `clips_dir`: Directory containing the extracted voice clips

**Returns:**
- The Hugging Face repository name

#### `train_model(dataset_repo: str) -> Path`
Trains the TTS model using LoRA fine-tuning.

**Parameters:**
- `dataset_repo`: Hugging Face repository containing the training dataset

**Returns:**
- Path to the saved LoRA adapter

#### `generate_demo(lora_dir: Path, text: str, output_path: Optional[Path] = None) -> Path`
Generates demo audio using the fine-tuned model.

**Parameters:**
- `lora_dir`: Path to the LoRA adapter directory
- `text`: Text to synthesize
- `output_path`: Optional path for the output audio file

**Returns:**
- Path to the generated audio file

## Component Classes

### YouTubeDownloader

Downloads and processes audio from YouTube.

```python
class YouTubeDownloader:
    def __init__(self, config: PipelineConfig, logger: logging.Logger)
    def download(self) -> Path
```

### VoiceExtractor

Extracts specific speaker's voice using Voice_Extractor.

```python
class VoiceExtractor:
    def __init__(self, config: PipelineConfig, logger: logging.Logger)
    def extract_voice(self, source_wav: Path) -> Path
```

### DatasetPreparer

Prepares and uploads dataset to Hugging Face Hub.

```python
class DatasetPreparer:
    def __init__(self, config: PipelineConfig, logger: logging.Logger)
    def prepare_and_upload(self, dataset_dir: Path) -> str
```

### TTSTrainer

Fine-tunes TTS model using LoRA.

```python
class TTSTrainer:
    def __init__(self, config: PipelineConfig, logger: logging.Logger)
    def train(self, dataset_repo: str) -> Path
```

### SpeechSynthesizer

Generates speech using fine-tuned model.

```python
class SpeechSynthesizer:
    def __init__(self, logger: logging.Logger)
    def generate_demo(self, lora_dir: Path, base_model: str, text: str, output_path: Optional[Path] = None) -> Path
```

## Utility Functions

### setup_logging

Sets up logging configuration.

```python
def setup_logging(level: str = "INFO", format_str: str = "...") -> logging.Logger
```

**Parameters:**
- `level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `format_str`: Log message format string

**Returns:**
- Configured logger instance

### run_command

Executes shell commands with error handling.

```python
def run_command(cmd: List[str], logger: logging.Logger, **kwargs: Any) -> None
```

**Parameters:**
- `cmd`: Command to execute as list of strings
- `logger`: Logger instance for output
- `**kwargs`: Additional arguments passed to subprocess.call

**Raises:**
- `SystemExit`: If command returns non-zero exit code

## CLI Commands

### run

Runs the complete voice cloning pipeline.

```bash
voice-clone run <youtube_url> [OPTIONS]
```

**Options:**
- `--output-dir, -o`: Output directory for generated files
- `--hf-repo, -r`: Hugging Face repository name for dataset
- `--max-steps, -s`: Maximum training steps
- `--demo-text, -t`: Text to synthesize for demo
- `--ref-start`: Start time (seconds) for reference audio slice
- `--ref-end`: End time (seconds) for reference audio slice
- `--verbose, -v`: Enable verbose logging

### download

Downloads audio from YouTube URL only.

```bash
voice-clone download <youtube_url> [OPTIONS]
```

**Options:**
- `--output-dir, -o`: Output directory for downloaded audio

### synthesize

Generates speech using a trained LoRA adapter.

```bash
voice-clone synthesize <lora_dir> [OPTIONS]
```

**Options:**
- `--text, -t`: Text to synthesize
- `--output, -o`: Output audio file path
- `--base-model`: Base TTS model to use

## Environment Variables

### Required
- `HF_TOKEN`: Hugging Face authentication token

### Optional Configuration
- `DEFAULT_OUTPUT_DIR`: Default output directory (default: "./voice_clone")
- `DEFAULT_HF_REPO`: Default HuggingFace repository (default: "username/female_voice_ds")
- `DEFAULT_BASE_TTS_MODEL`: Default base TTS model (default: "unsloth/sesame-csm-1b")
- `DEFAULT_MAX_STEPS`: Default training steps (default: "1000")
- `DEFAULT_REF_START`: Default reference start time (default: "0")
- `DEFAULT_REF_END`: Default reference end time (default: "15")

### Audio Processing
- `AUDIO_SAMPLE_RATE`: Audio sample rate for processing (default: "44100")
- `AUDIO_CHANNELS`: Number of audio channels (default: "2")
- `TTS_SAMPLE_RATE`: Sample rate for TTS processing (default: "24000")

### Training Configuration
- `DEFAULT_BATCH_SIZE`: Training batch size (default: "2")
- `DEFAULT_GRADIENT_ACCUMULATION`: Gradient accumulation steps (default: "8")
- `DEFAULT_LEARNING_RATE`: Learning rate (default: "2e-4")

### Logging
- `LOG_LEVEL`: Logging level (default: "INFO")
- `LOG_FORMAT`: Log message format (default: "%(levelname)s | %(name)s | %(message)s")

## Error Handling

The pipeline uses consistent error handling:

### Configuration Errors
Raised when configuration is invalid:
```python
raise ValueError("HF_TOKEN is required. Please set it in your .env file or environment.")
```

### System Errors
For critical failures that should stop execution:
```python
logger.error("Command failed with exit code %d", result)
sys.exit(1)
```

### Import Errors
When required dependencies are missing:
```python
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install package_name")
    sys.exit(1)
```

## Examples

### Basic Usage

```python
from voice_pipeline import VoiceCloningPipeline, PipelineConfig

# Simple configuration
config = PipelineConfig(yt_url="https://www.youtube.com/watch?v=example")

# Run complete pipeline
pipeline = VoiceCloningPipeline(config)
demo_path = pipeline.run_full_pipeline()
print(f"Demo generated: {demo_path}")
```

### Advanced Configuration

```python
from pathlib import Path
from voice_pipeline import VoiceCloningPipeline, PipelineConfig, LoggingConfig

# Advanced configuration
config = PipelineConfig(
    yt_url="https://www.youtube.com/watch?v=example",
    out_dir=Path("custom_output"),
    hf_repo="myuser/custom_dataset",
    base_tts="custom/tts-model",
    max_steps=2000,
    ref_seconds=(30, 60),  # Use 30-60 seconds as reference
    batch_size=4,
    learning_rate=1e-4
)

# Custom logging
logging_config = LoggingConfig(level="DEBUG")

# Initialize and run
pipeline = VoiceCloningPipeline(config, logging_config)
demo_path = pipeline.run_full_pipeline("Custom demo text to synthesize")
```

### Individual Components

```python
# Use individual components
pipeline = VoiceCloningPipeline(config)

# Step by step execution
audio_path = pipeline.download_audio()
print(f"Audio downloaded to: {audio_path}")

clips_dir = pipeline.extract_voice(audio_path)
print(f"Voice extracted to: {clips_dir}")

dataset_repo = pipeline.prepare_dataset(clips_dir)
print(f"Dataset uploaded to: {dataset_repo}")

lora_dir = pipeline.train_model(dataset_repo)
print(f"Model trained, adapter saved to: {lora_dir}")

demo_path = pipeline.generate_demo(lora_dir, "Hello from the cloned voice!")
print(f"Demo generated: {demo_path}")
```
