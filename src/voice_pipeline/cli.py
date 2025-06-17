#!/usr/bin/env python3
"""
Command-line interface for the voice cloning pipeline.
"""

import sys
import click
from pathlib import Path

from voice_pipeline import VoiceCloningPipeline, PipelineConfig
from voice_pipeline.config import LoggingConfig


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Voice Cloning Pipeline - Clone voices from YouTube videos."""
    pass


@cli.command()
@click.argument("youtube_url")
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for generated files"
)
@click.option(
    "--hf-repo", "-r",
    help="Hugging Face repository name for dataset"
)
@click.option(
    "--max-steps", "-s",
    type=int,
    help="Maximum training steps"
)
@click.option(
    "--demo-text", "-t",
    default="Hello, world! This is a demo of the cloned voice.",
    help="Text to synthesize for demo"
)
@click.option(
    "--ref-start",
    type=int,
    help="Start time (seconds) for reference audio slice"
)
@click.option(
    "--ref-end",
    type=int,
    help="End time (seconds) for reference audio slice"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def run(youtube_url, output_dir, hf_repo, max_steps, demo_text, ref_start, ref_end, verbose):
    """Run the complete voice cloning pipeline."""
    
    # Build configuration
    config_kwargs = {"yt_url": youtube_url}
    
    if output_dir:
        config_kwargs["out_dir"] = output_dir
    if hf_repo:
        config_kwargs["hf_repo"] = hf_repo
    if max_steps:
        config_kwargs["max_steps"] = max_steps
    if ref_start is not None and ref_end is not None:
        config_kwargs["ref_seconds"] = (ref_start, ref_end)
    
    try:
        config = PipelineConfig(**config_kwargs)
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    
    # Setup logging
    logging_config = LoggingConfig(level="DEBUG" if verbose else "INFO")
    
    # Run pipeline
    try:
        pipeline = VoiceCloningPipeline(config, logging_config)
        demo_path = pipeline.run_full_pipeline(demo_text)
        
        click.echo(f"\n✅ Voice cloning completed successfully!")
        click.echo(f"Demo audio: {demo_path}")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("youtube_url")
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for downloaded audio"
)
def download(youtube_url, output_dir):
    """Download audio from YouTube URL only."""
    
    config_kwargs = {"yt_url": youtube_url}
    if output_dir:
        config_kwargs["out_dir"] = output_dir
        
    try:
        config = PipelineConfig(**config_kwargs)
        pipeline = VoiceCloningPipeline(config)
        wav_path = pipeline.download_audio()
        
        click.echo(f"✅ Audio downloaded: {wav_path}")
        
    except Exception as e:
        click.echo(f"❌ Download failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("lora_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--text", "-t",
    default="Hello, world! This is a demo of the cloned voice.",
    help="Text to synthesize"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output audio file path"
)
@click.option(
    "--base-model",
    default="unsloth/sesame-csm-1b",
    help="Base TTS model to use"
)
def synthesize(lora_dir, text, output, base_model):
    """Generate speech using a trained LoRA adapter."""
    
    from voice_pipeline.synthesizer import SpeechSynthesizer
    from voice_pipeline.utils import setup_logging
    
    logger = setup_logging()
    synthesizer = SpeechSynthesizer(logger)
    
    try:
        audio_path = synthesizer.generate_demo(
            lora_dir, base_model, text, output
        )
        click.echo(f"✅ Audio generated: {audio_path}")
        
    except Exception as e:
        click.echo(f"❌ Synthesis failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
