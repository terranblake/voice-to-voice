"""
Voice Cloning Pipeline
~~~~~~~~~~~~~~~~~~~~~

A comprehensive toolkit for voice cloning using YouTube audio sources.
"""

__version__ = "0.1.0"
__author__ = "Voice Pipeline Team"

# Import main classes for public API
try:
    from .pipeline import VoiceCloningPipeline
    from .config import PipelineConfig, LoggingConfig
    
    __all__ = ["VoiceCloningPipeline", "PipelineConfig", "LoggingConfig"]
except ImportError:
    # Handle case where dependencies might not be installed
    __all__ = []
