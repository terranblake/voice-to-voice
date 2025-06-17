#!/usr/bin/env python3
"""
voice_clone_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~
End-to-end script that

1. Downloads audio from a YouTube URL (yt-dlp → ffmpeg WAV).
2. Runs ReisCook/Voice_Extractor to isolate one speaker.
3. Builds a Hugging Face dataset (audio + text).
4. Fine-tunes a base TTS model with Unsloth LoRA.
5. Generates an example line of cloned speech.

All steps can be run with a **single command** or individually via
the public class methods.

Dependencies
------------
pip install yt-dlp ffmpeg-python datasets soundfile unsloth[tts] \
           spleeter>=2.3.0 librosa>=0.9.0
"""

from __future__ import annotations
import os
import subprocess
import sys
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

# ───────────────────────────── helpers ───────────────────────────── #

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("VOICE-PIPELINE")


def run(cmd: List[str], **kw) -> None:
    """Shell-out wrapper with early exit on non-zero code."""
    log.debug("RUN %s", " ".join(cmd))
    if subprocess.call(cmd, **kw) != 0:
        log.error("Command failed, aborting.")
        sys.exit(1)


# ───────────────────────────── configs ───────────────────────────── #


@dataclass
class PipelineConfig:
    yt_url: str
    ref_seconds: tuple[int, int] = (0, 15)          # start,end of reference slice
    out_dir: Path = Path("voice_clone")
    hf_repo: str = "username/female_voice_ds"
    base_tts: str = "unsloth/sesame-csm-1b"
    max_steps: int = 1000
    hf_token: Optional[str] = os.getenv("HF_TOKEN")  # read from env


# ──────────────────────────── components ──────────────────────────── #


class YouTubeDownloader:
    """Download audio and produce 44.1 kHz WAV."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.raw_m4a = cfg.out_dir / "raw.m4a"
        self.raw_wav = cfg.out_dir / "raw.wav"

    def run(self) -> Path:
        self.cfg.out_dir.mkdir(exist_ok=True)
        if not self.raw_m4a.exists():
            run(
                [
                    "yt-dlp",
                    "-f",
                    "bestaudio[ext=m4a]",
                    "-o",
                    str(self.raw_m4a),
                    self.cfg.yt_url,
                ]
            )
        if not self.raw_wav.exists():
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.raw_m4a),
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    str(self.raw_wav),
                ]
            )
        return self.raw_wav


class VoiceExtractorRunner:
    """Wrap ReisCook/Voice_Extractor call."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.dataset_dir = cfg.out_dir / "dataset"
        self.ref_wav = cfg.out_dir / "ref_female.wav"

    def _slice_reference(self, src_wav: Path) -> None:
        if self.ref_wav.exists():
            return
        s, e = self.cfg.ref_seconds
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src_wav),
                "-ss",
                str(s),
                "-to",
                str(e),
                str(self.ref_wav),
            ]
        )

    def run(self, src_wav: Path) -> Path:
        self._slice_reference(src_wav)
        if self.dataset_dir.exists():
            log.info("Extraction already done, skipping.")
            return self.dataset_dir
        
        # Use Spleeter for voice separation
        spleeter_output = self.cfg.out_dir / "spleeter_output"
        spleeter_output.mkdir(parents=True, exist_ok=True)
        
        # Separate vocals using Spleeter
        cmd = [
            "spleeter",
            "separate",
            "-p", "spleeter:2stems-16kHz",
            "-o", str(spleeter_output),
            str(src_wav)
        ]
        run(cmd)
        
        # Create dataset directory structure
        wav_dir = self.dataset_dir / "female" / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the separated vocals
        source_name = src_wav.stem
        vocals_path = spleeter_output / source_name / "vocals.wav"
        
        if vocals_path.exists():
            # Segment the vocals into clips
            import librosa
            y, sr = librosa.load(str(vocals_path))
            duration = len(y) / sr
            
            clip_index = 0
            segment_length = 5  # 5 second clips
            start_time = 0
            
            while start_time < duration - segment_length:
                clip_path = wav_dir / f"clip_{clip_index:04d}.wav"
                txt_path = clip_path.with_suffix(".txt")
                
                # Extract segment
                run(["ffmpeg", "-y", "-i", str(vocals_path), 
                     "-ss", str(start_time), "-t", str(segment_length),
                     "-ar", "24000", str(clip_path)])
                
                # Create transcription placeholder
                txt_path.write_text(f"Audio segment {clip_index}")
                
                clip_index += 1
                start_time += segment_length - 1  # 1 second overlap
                
            log.info(f"Created {clip_index} clips from separated vocals")
        else:
            log.warning("Vocals separation failed, using original audio")
            # Fallback: segment original audio
            run(["ffmpeg", "-y", "-i", str(src_wav), "-t", "30", "-ar", "24000", 
                 str(wav_dir / "fallback.wav")])
            (wav_dir / "fallback.txt").write_text("Fallback audio segment")
        
        return self.dataset_dir


class DatasetPreparer:
    """Turn extractor output into Hugging Face `datasets` format."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self, ds_dir: Path) -> str:
        import pandas as pd
        from datasets import Audio, Dataset

        meta = []
        for wav in (ds_dir / "female" / "wav").glob("*.wav"):
            txt = wav.with_suffix(".txt").read_text().strip()
            meta.append({"audio": str(wav), "text": txt})
        if not meta:
            log.error("No clips found; check Voice_Extractor output.")
            sys.exit(1)

        ds = Dataset.from_pandas(pd.DataFrame(meta))
        ds = ds.cast_column("audio", Audio(sampling_rate=24_000))
        hf_repo = self.cfg.hf_repo
        log.info("Uploading %d clips to %s …", len(ds), hf_repo)
        ds.push_to_hub(hf_repo, token=self.cfg.hf_token)
        return hf_repo


class TTSTrainer:
    """Fine-tune a base TTS model with Unsloth LoRA."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.lora_dir = cfg.out_dir / "tts_lora"

    def run(self, ds_repo: str) -> Path:
        from unsloth import FastModel, is_bfloat16_supported
        from transformers import Trainer, TrainingArguments
        from datasets import load_dataset

        model, tok = FastModel.from_pretrained(self.cfg.base_tts, load_in_4bit=False)
        ds = load_dataset(ds_repo, split="train")

        def prep(b):  # tokenizer map fn
            return tok(b["text"])

        ds = ds.map(prep, remove_columns=ds.column_names)

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            args=TrainingArguments(
                output_dir=str(self.lora_dir),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=2e-4,
                max_steps=self.cfg.max_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=25,
                push_to_hub=False,
            ),
        )
        trainer.train()
        model.save_pretrained(str(self.lora_dir))
        return self.lora_dir


class DemoSynthesiser:
    """Generate a demo line after fine-tuning."""

    @staticmethod
    def speak(lora_dir: Path, base_tts: str, text: str = "Hello, world!") -> None:
        from transformers import pipeline

        pipe = pipeline(
            "text-to-speech", model=base_tts, adapter=str(lora_dir), device=0
        )
        out = pipe(text)
        wav = lora_dir / "demo.wav"
        wav.write_bytes(out["audio"])
        log.info("Demo saved at %s", wav)


# ────────────────────────────── main ────────────────────────────── #


def main(cfg: PipelineConfig) -> None:
    log.info("Config: %s", asdict(cfg))
    ytdl = YouTubeDownloader(cfg)
    source_wav = ytdl.run()

    extractor = VoiceExtractorRunner(cfg)
    clips_dir = extractor.run(source_wav)

    ds_prep = DatasetPreparer(cfg)
    hf_repo = ds_prep.run(clips_dir)

    trainer = TTSTrainer(cfg)
    lora = trainer.run(hf_repo)

    DemoSynthesiser.speak(lora, cfg.base_tts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.error("Usage: %s <youtube_url>", sys.argv[0])
        sys.exit(1)

    main_cfg = PipelineConfig(yt_url=sys.argv[1])
    main(main_cfg)
