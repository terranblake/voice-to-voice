"""
YouTube audio downloader component.
"""

import logging
from pathlib import Path

from .utils import run_command
from .config import PipelineConfig


class YouTubeDownloader:
    """Download audio from YouTube and convert to WAV format."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.raw_m4a = config.out_dir / "raw.m4a"
        self.raw_wav = config.out_dir / "raw.wav"

    def download(self) -> Path:
        """
        Download audio from YouTube URL and convert to WAV.
        
        Returns:
            Path to the converted WAV file
        """
        self.config.out_dir.mkdir(exist_ok=True)
        
        # Download audio if not already present
        if not self.raw_m4a.exists():
            self.logger.info(f"Downloading audio from: {self.config.yt_url}")
            run_command(
                [
                    "yt-dlp",
                    "-f",
                    "bestaudio[ext=m4a]",
                    "-o",
                    str(self.raw_m4a),
                    self.config.yt_url,
                ],
                self.logger
            )
        else:
            self.logger.info("Audio file already exists, skipping download")

        # Convert to WAV if not already present
        if not self.raw_wav.exists():
            self.logger.info("Converting audio to WAV format")
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.raw_m4a),
                    "-ar",
                    str(self.config.audio_sample_rate),
                    "-ac",
                    str(self.config.audio_channels),
                    str(self.raw_wav),
                ],
                self.logger
            )
        else:
            self.logger.info("WAV file already exists, skipping conversion")

        return self.raw_wav
