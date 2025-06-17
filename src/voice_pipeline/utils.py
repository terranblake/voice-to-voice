"""
Utility functions for the voice cloning pipeline.
"""

import logging
import subprocess
import sys
from typing import List, Any


def setup_logging(level: str = "INFO", format_str: str = "%(levelname)s | %(name)s | %(message)s") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(level=getattr(logging, level.upper()), format=format_str)
    return logging.getLogger("VOICE-PIPELINE")


def run_command(cmd: List[str], logger: logging.Logger, **kwargs: Any) -> None:
    """
    Execute a shell command with error handling.
    
    Args:
        cmd: Command to execute as list of strings
        logger: Logger instance for output
        **kwargs: Additional arguments passed to subprocess.call
        
    Raises:
        SystemExit: If command returns non-zero exit code
    """
    logger.debug("Executing command: %s", " ".join(cmd))
    
    try:
        result = subprocess.call(cmd, **kwargs)
        if result != 0:
            logger.error("Command failed with exit code %d: %s", result, " ".join(cmd))
            sys.exit(1)
    except FileNotFoundError as e:
        logger.error("Command not found: %s", e)
        logger.error("Please ensure all required dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error running command: %s", e)
        sys.exit(1)
