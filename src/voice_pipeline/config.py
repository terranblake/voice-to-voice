"""
Configuration module for the voice cloning pipeline.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PipelineConfig:
    """Configuration for the voice cloning pipeline."""
    
    # Required
    yt_url: str
    
    # Audio processing
    ref_seconds: Tuple[int, int] = field(
        default_factory=lambda: (
            int(os.getenv("DEFAULT_REF_START", "0")),
            int(os.getenv("DEFAULT_REF_END", "15"))
        )
    )
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
    audio_channels: int = int(os.getenv("AUDIO_CHANNELS", "2"))
    tts_sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    
    # Directories and repositories
    out_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DEFAULT_OUTPUT_DIR", "voice_clone"))
    )
    hf_repo: str = os.getenv("DEFAULT_HF_REPO", "username/female_voice_ds")
    
    # Model configuration
    base_tts: str = os.getenv("DEFAULT_BASE_TTS_MODEL", "unsloth/sesame-csm-1b")
    
    # Training configuration
    max_steps: int = int(os.getenv("DEFAULT_MAX_STEPS", "1000"))
    batch_size: int = int(os.getenv("DEFAULT_BATCH_SIZE", "2"))
    gradient_accumulation_steps: int = int(os.getenv("DEFAULT_GRADIENT_ACCUMULATION", "8"))
    learning_rate: float = float(os.getenv("DEFAULT_LEARNING_RATE", "2e-4"))
    
    # Authentication
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required. Please set it in your .env file or environment."
            )
        
        if not self.yt_url:
            raise ValueError("YouTube URL is required.")
        
        # Ensure output directory is Path object
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "%(levelname)s | %(name)s | %(message)s")
