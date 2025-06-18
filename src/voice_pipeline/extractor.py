"""
Voice extraction component using ReisCook/Voice_Extractor.
"""

import logging
from pathlib import Path

from .utils import run_command
from .config import PipelineConfig


class VoiceExtractor:
    """Extract and isolate a specific speaker's voice using Voice_Extractor."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dataset_dir = config.out_dir / "dataset"
        self.ref_wav = config.out_dir / "ref_female.wav"

    def _create_reference_slice(self, source_wav: Path) -> None:
        """
        Create a reference audio slice from the source audio.
        
        Args:
            source_wav: Path to the source WAV file
        """
        if self.ref_wav.exists():
            self.logger.info("Reference audio slice already exists")
            return

        start_sec, end_sec = self.config.ref_seconds
        self.logger.info(f"Creating reference slice from {start_sec}s to {end_sec}s")
        
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_wav),
                "-ss",
                str(start_sec),
                "-to",
                str(end_sec),
                str(self.ref_wav),
            ],
            self.logger
        )

    def extract_voice(self, source_wav: Path) -> Path:
        """
        Extract the target voice from the source audio.
        
        Args:
            source_wav: Path to the source WAV file
            
        Returns:
            Path to the directory containing extracted voice clips
        """
        self._create_reference_slice(source_wav)

        if self.dataset_dir.exists() and any(self.dataset_dir.rglob("*.wav")):
            self.logger.info("Voice extraction already completed, skipping")
            return self.dataset_dir

        self.logger.info("Starting voice extraction process")
        
        # Find the Voice_Extractor script in the submodule
        voice_extractor_path = Path(__file__).parent.parent.parent / "extern" / "Voice_Extractor" / "run_extractor.py"
        if not voice_extractor_path.exists():
            # Try relative to current working directory
            voice_extractor_path = Path.cwd() / "extern" / "Voice_Extractor" / "run_extractor.py"
            
        if not voice_extractor_path.exists():
            raise FileNotFoundError(
                "Voice_Extractor submodule not found. Please run 'git submodule update --init --recursive' "
                "to initialize the submodule."
            )
        
        cmd = [
            "python",
            str(voice_extractor_path),
            "--input-audio",
            str(source_wav),
            "--reference-audio",
            str(self.ref_wav),
            "--target-name",
            "female",
            "--output-base-dir",
            str(self.dataset_dir),
        ]
            
        run_command(cmd, self.logger)
        
        self.logger.info(f"Voice extraction completed. Output saved to: {self.dataset_dir}")
        return self.dataset_dir
