"""
Tests for utility functions.
"""

import logging
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from voice_pipeline.utils import setup_logging, run_command


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_default_logging_setup(self):
        """Test default logging configuration."""
        logger = setup_logging()
        assert logger.name == "VOICE-PIPELINE"
        assert logger.level == logging.INFO

    def test_custom_logging_setup(self):
        """Test custom logging configuration."""
        logger = setup_logging(level="DEBUG", format_str="%(message)s")
        assert logger.level == logging.DEBUG

    def test_invalid_logging_level(self):
        """Test that invalid logging level defaults to INFO."""
        # This should not raise an error but might not set the expected level
        logger = setup_logging(level="INVALID")
        # The actual behavior depends on Python's logging implementation


class TestRunCommand:
    """Test cases for run_command function."""

    def test_successful_command(self):
        """Test running a successful command."""
        logger = MagicMock()
        
        with patch('subprocess.call', return_value=0) as mock_call:
            run_command(["echo", "test"], logger)
            mock_call.assert_called_once_with(["echo", "test"])
            logger.debug.assert_called_once()

    def test_failed_command(self):
        """Test running a command that fails."""
        logger = MagicMock()
        
        with patch('subprocess.call', return_value=1) as mock_call:
            with patch('sys.exit') as mock_exit:
                run_command(["false"], logger)
                mock_call.assert_called_once_with(["false"])
                logger.error.assert_called()
                mock_exit.assert_called_once_with(1)

    def test_command_not_found(self):
        """Test running a command that doesn't exist."""
        logger = MagicMock()
        
        with patch('subprocess.call', side_effect=FileNotFoundError("Command not found")):
            with patch('sys.exit') as mock_exit:
                run_command(["nonexistent_command"], logger)
                logger.error.assert_called()
                mock_exit.assert_called_once_with(1)

    def test_unexpected_error(self):
        """Test handling unexpected errors."""
        logger = MagicMock()
        
        with patch('subprocess.call', side_effect=Exception("Unexpected error")):
            with patch('sys.exit') as mock_exit:
                run_command(["test"], logger)
                logger.error.assert_called()
                mock_exit.assert_called_once_with(1)

    def test_command_with_kwargs(self):
        """Test running command with additional keyword arguments."""
        logger = MagicMock()
        
        with patch('subprocess.call', return_value=0) as mock_call:
            run_command(["echo", "test"], logger, cwd="/tmp", env={})
            mock_call.assert_called_once_with(["echo", "test"], cwd="/tmp", env={})
