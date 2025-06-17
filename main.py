#!/usr/bin/env python3
"""
Legacy main script for backward compatibility.
This script provides the same interface as the original example.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.voice_pipeline import VoiceCloningPipeline, PipelineConfig


def main():
    """Main entry point matching the original script interface."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube_url>", file=sys.stderr)
        print("       python -m voice_pipeline.cli run <youtube_url>  # Preferred CLI")
        sys.exit(1)

    try:
        config = PipelineConfig(yt_url=sys.argv[1])
        pipeline = VoiceCloningPipeline(config)
        demo_path = pipeline.run_full_pipeline()
        
        print(f"\n✅ Voice cloning completed successfully!")
        print(f"Demo audio: {demo_path}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
