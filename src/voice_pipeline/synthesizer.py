"""
Speech synthesis component for generating demo audio.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class SpeechSynthesizer:
    """Generate speech using the fine-tuned TTS model."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def generate_demo(
        self, 
        lora_dir: Path, 
        base_model: str, 
        text: str = "Hello, world! This is a demo of the cloned voice.",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a demo audio file using the fine-tuned model.
        
        Args:
            lora_dir: Path to the LoRA adapter directory
            base_model: Name of the base TTS model
            text: Text to synthesize
            output_path: Optional path for the output audio file
            
        Returns:
            Path to the generated audio file
        """
        try:
            from transformers import pipeline
        except ImportError as e:
            self.logger.error(f"Required packages not installed: {e}")
            self.logger.error("Please install: pip install transformers")
            sys.exit(1)

        if output_path is None:
            output_path = lora_dir / "demo.wav"

        self.logger.info(f"Generating demo audio with text: '{text}'")
        
        try:
            # Create TTS pipeline with the fine-tuned adapter
            tts_pipeline = pipeline(
                "text-to-speech",
                model=base_model,
                adapter=str(lora_dir),
                device=0 if self._is_gpu_available() else -1
            )
            
            # Generate audio
            result = tts_pipeline(text)
            
            # Save audio to file
            with open(output_path, "wb") as f:
                f.write(result["audio"])
                
            self.logger.info(f"Demo audio saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate demo audio: {e}")
            # Fallback: try without GPU
            if self._is_gpu_available():
                self.logger.info("Retrying without GPU...")
                try:
                    tts_pipeline = pipeline(
                        "text-to-speech",
                        model=base_model,
                        adapter=str(lora_dir),
                        device=-1
                    )
                    result = tts_pipeline(text)
                    with open(output_path, "wb") as f:
                        f.write(result["audio"])
                    self.logger.info(f"Demo audio saved to: {output_path}")
                except Exception as e2:
                    self.logger.error(f"Failed to generate demo audio even without GPU: {e2}")
                    sys.exit(1)
            else:
                sys.exit(1)

        return output_path

    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for inference."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
