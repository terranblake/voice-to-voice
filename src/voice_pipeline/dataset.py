"""
Dataset preparation component for Hugging Face datasets.
"""

import logging
import sys
from pathlib import Path

from .config import PipelineConfig


class DatasetPreparer:
    """Prepare and upload dataset to Hugging Face Hub."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def prepare_and_upload(self, dataset_dir: Path) -> str:
        """
        Prepare dataset from extracted voice clips and upload to Hugging Face Hub.
        
        Args:
            dataset_dir: Directory containing the extracted voice clips
            
        Returns:
            The Hugging Face repository name
        """
        try:
            import pandas as pd
            from datasets import Audio, Dataset
        except ImportError as e:
            self.logger.error(f"Required packages not installed: {e}")
            self.logger.error("Please install: pip install datasets pandas")
            sys.exit(1)

        # Collect audio files and their transcriptions
        wav_dir = dataset_dir / "female" / "wav"
        if not wav_dir.exists():
            self.logger.error(f"Expected WAV directory not found: {wav_dir}")
            sys.exit(1)

        metadata = []
        wav_files = list(wav_dir.glob("*.wav"))
        
        if not wav_files:
            self.logger.error("No WAV files found in the dataset directory")
            sys.exit(1)

        self.logger.info(f"Found {len(wav_files)} audio clips")

        for wav_file in wav_files:
            txt_file = wav_file.with_suffix(".txt")
            if txt_file.exists():
                text_content = txt_file.read_text().strip()
                metadata.append({
                    "audio": str(wav_file),
                    "text": text_content
                })
            else:
                self.logger.warning(f"Missing transcript for: {wav_file}")

        if not metadata:
            self.logger.error("No valid audio-text pairs found")
            sys.exit(1)

        self.logger.info(f"Preparing dataset with {len(metadata)} samples")

        # Create Hugging Face dataset
        df = pd.DataFrame(metadata)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.tts_sample_rate))

        # Upload to Hugging Face Hub
        repo_name = self.config.hf_repo
        self.logger.info(f"Uploading dataset to Hugging Face Hub: {repo_name}")
        
        try:
            dataset.push_to_hub(repo_name, token=self.config.hf_token)
            self.logger.info("Dataset upload completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to upload dataset: {e}")
            sys.exit(1)

        return repo_name
