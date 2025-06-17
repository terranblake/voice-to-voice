"""
Tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.voice_pipeline.config import PipelineConfig, LoggingConfig


class TestPipelineConfig:
    """Test cases for PipelineConfig."""

    def test_minimal_config(self):
        """Test creating config with minimal required parameters."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            config = PipelineConfig(yt_url="https://youtube.com/watch?v=test")
            assert config.yt_url == "https://youtube.com/watch?v=test"
            assert config.hf_token == "test_token"
            assert isinstance(config.out_dir, Path)

    def test_missing_hf_token(self):
        """Test that missing HF_TOKEN raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HF_TOKEN is required"):
                PipelineConfig(yt_url="https://youtube.com/watch?v=test")

    def test_missing_yt_url(self):
        """Test that missing yt_url raises ValueError."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with pytest.raises(ValueError, match="YouTube URL is required"):
                PipelineConfig(yt_url="")

    def test_custom_config(self):
        """Test creating config with custom parameters."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            config = PipelineConfig(
                yt_url="https://youtube.com/watch?v=test",
                out_dir=Path("custom_dir"),
                hf_repo="test/repo",
                max_steps=500
            )
            assert config.out_dir == Path("custom_dir")
            assert config.hf_repo == "test/repo"
            assert config.max_steps == 500

    def test_environment_variable_defaults(self):
        """Test that environment variables are used as defaults."""
        env_vars = {
            "HF_TOKEN": "test_token",
            "DEFAULT_OUTPUT_DIR": "env_output",
            "DEFAULT_HF_REPO": "env/repo",
            "DEFAULT_MAX_STEPS": "2000"
        }
        with patch.dict(os.environ, env_vars):
            config = PipelineConfig(yt_url="https://youtube.com/watch?v=test")
            assert config.out_dir == Path("env_output")
            assert config.hf_repo == "env/repo"
            assert config.max_steps == 2000


class TestLoggingConfig:
    """Test cases for LoggingConfig."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(levelname)s" in config.format

    def test_custom_logging_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(level="DEBUG", format="%(message)s")
        assert config.level == "DEBUG"
        assert config.format == "%(message)s"

    def test_environment_logging_config(self):
        """Test logging config from environment variables."""
        env_vars = {
            "LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "%(name)s: %(message)s"
        }
        with patch.dict(os.environ, env_vars):
            config = LoggingConfig()
            assert config.level == "WARNING"
            assert config.format == "%(name)s: %(message)s"
