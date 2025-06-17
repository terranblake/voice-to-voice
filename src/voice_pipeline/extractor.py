"""
Voice extraction component using Spleeter for audio source separation.
"""

import logging
import os
import sys
from pathlib import Path
import json

from .utils import run_command
from .config import PipelineConfig


class VoiceExtractor:
    """Extract and isolate vocals from audio using Spleeter."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dataset_dir = config.out_dir / "dataset"
        self.ref_wav = config.out_dir / "ref_female.wav"
        self.spleeter_output = config.out_dir / "spleeter_output"

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

    def _separate_vocals(self, source_wav: Path) -> Path:
        """
        Use Spleeter to separate vocals from the audio.
        
        Args:
            source_wav: Path to the source WAV file
            
        Returns:
            Path to the separated vocals file
        """
        self.spleeter_output.mkdir(parents=True, exist_ok=True)
        
        # Use 2stems model (vocals/accompaniment)
        self.logger.info("Separating vocals using Spleeter...")
        
        run_command([
            "spleeter",
            "separate",
            "-p", "spleeter:2stems-16kHz",
            "-o", str(self.spleeter_output),
            str(source_wav)
        ], self.logger)
        
        # The vocals will be in spleeter_output/<filename>/vocals.wav
        source_name = source_wav.stem
        vocals_path = self.spleeter_output / source_name / "vocals.wav"
        
        if not vocals_path.exists():
            raise FileNotFoundError(f"Expected vocals file not found: {vocals_path}")
            
        return vocals_path

    def _segment_audio(self, vocals_path: Path) -> None:
        """
        Segment the vocals audio into smaller clips for training.
        
        Args:
            vocals_path: Path to the separated vocals file
        """
        # Create dataset structure similar to voice_extractor output
        wav_dir = self.dataset_dir / "female" / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Segmenting vocals into training clips...")
        
        # Segment into 5-second clips with 1-second overlap
        segment_length = 5
        overlap = 1
        
        # Get audio duration first
        try:
            import librosa
            y, sr = librosa.load(str(vocals_path))
            duration = len(y) / sr
        except ImportError:
            self.logger.error("librosa is required for audio processing")
            sys.exit(1)
        
        clip_index = 0
        start_time = 0
        
        while start_time < duration - segment_length:
            end_time = start_time + segment_length
            
            clip_path = wav_dir / f"clip_{clip_index:04d}.wav"
            txt_path = clip_path.with_suffix(".txt")
            
            # Extract segment using FFmpeg
            run_command([
                "ffmpeg", "-y",
                "-i", str(vocals_path),
                "-ss", str(start_time),
                "-t", str(segment_length),
                "-ar", "24000",  # Resample to 24kHz for TTS
                str(clip_path)
            ], self.logger)
            
            # Create placeholder transcription 
            # In a real scenario, you'd use a speech recognition model
            txt_path.write_text(f"Audio segment {clip_index}")
            
            clip_index += 1
            start_time += segment_length - overlap
        
        self.logger.info(f"Created {clip_index} audio clips in {wav_dir}")

    def extract_voice(self, source_wav: Path) -> Path:
        """
        Extract vocals from the source audio using Spleeter.
        
        Args:
            source_wav: Path to the source WAV file
            
        Returns:
            Path to the directory containing extracted voice clips
        """
        self._create_reference_slice(source_wav)

        if self.dataset_dir.exists() and any(self.dataset_dir.rglob("*.wav")):
            self.logger.info("Voice extraction already completed, skipping")
            return self.dataset_dir

        self.logger.info("Starting voice extraction process with Spleeter")
        
        try:
            # Step 1: Separate vocals from music
            vocals_path = self._separate_vocals(source_wav)
            
            # Step 2: Segment vocals into training clips
            self._segment_audio(vocals_path)
            
        except Exception as e:
            self.logger.error(f"Voice extraction failed: {e}")
            # Fallback: Just segment the original audio
            self.logger.info("Falling back to segmenting original audio...")
            self._segment_audio(source_wav)
        
        self.logger.info(f"Voice extraction completed. Output saved to: {self.dataset_dir}")
        return self.dataset_dir
