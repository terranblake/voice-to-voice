"""
Main voice cloning pipeline orchestrating all components.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, LoggingConfig
from .utils import setup_logging
from .downloader import YouTubeDownloader
from .extractor import VoiceExtractor
from .dataset import DatasetPreparer
from .trainer import TTSTrainer
from .synthesizer import SpeechSynthesizer


class VoiceCloningPipeline:
    """
    Main pipeline class that orchestrates the entire voice cloning process.
    
    This class provides both a one-shot execution method and individual
    component methods for fine-grained control.
    """

    def __init__(self, config: PipelineConfig, logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the voice cloning pipeline.
        
        Args:
            config: Pipeline configuration
            logging_config: Optional logging configuration
        """
        self.config = config
        
        # Setup logging
        if logging_config is None:
            logging_config = LoggingConfig()
        self.logger = setup_logging(logging_config.level, logging_config.format)
        
        # Initialize components
        self.downloader = YouTubeDownloader(config, self.logger)
        self.extractor = VoiceExtractor(config, self.logger)
        self.dataset_preparer = DatasetPreparer(config, self.logger)
        self.trainer = TTSTrainer(config, self.logger)
        self.synthesizer = SpeechSynthesizer(self.logger)

    def run_full_pipeline(self, demo_text: str = "Hello, world! This is a demo of the cloned voice.") -> Path:
        """
        Execute the complete voice cloning pipeline.
        
        Args:
            demo_text: Text to synthesize for the demo
            
        Returns:
            Path to the generated demo audio file
        """
        self.logger.info("Starting voice cloning pipeline")
        self.logger.info("Configuration: %s", asdict(self.config))

        # Step 1: Download audio from YouTube
        self.logger.info("Step 1/5: Downloading audio from YouTube")
        source_wav = self.download_audio()

        # Step 2: Extract target voice
        self.logger.info("Step 2/5: Extracting target voice")
        clips_dir = self.extract_voice(source_wav)

        # Step 3: Prepare dataset locally
        self.logger.info("Step 3/5: Preparing dataset locally")
        dataset_path = self.prepare_dataset(clips_dir)

        # Step 4: Train TTS model
        self.logger.info("Step 4/5: Training TTS model")
        lora_dir = self.train_model(dataset_path)

        # Step 5: Generate demo
        self.logger.info("Step 5/5: Generating demo audio")
        demo_path = self.generate_demo(lora_dir, demo_text)

        self.logger.info("Voice cloning pipeline completed successfully!")
        self.logger.info("Demo audio available at: %s", demo_path)
        
        return demo_path

    def download_audio(self) -> Path:
        """Download audio from YouTube URL."""
        return self.downloader.download()

    def extract_voice(self, source_wav: Path) -> Path:
        """Extract target voice from the source audio."""
        return self.extractor.extract_voice(source_wav)

    def prepare_dataset(self, clips_dir: Path) -> str:
        """Prepare dataset locally."""
        return self.dataset_preparer.prepare_and_save_local(clips_dir)

    def train_model(self, dataset_repo: str) -> Path:
        """Train the TTS model using LoRA fine-tuning."""
        return self.trainer.train(dataset_repo)

    def generate_demo(self, lora_dir: Path, text: str, output_path: Optional[Path] = None) -> Path:
        """Generate demo audio using the fine-tuned model."""
        return self.synthesizer.generate_demo(
            lora_dir, 
            self.config.base_tts, 
            text, 
            output_path
        )
