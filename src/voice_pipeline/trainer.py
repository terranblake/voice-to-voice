"""
TTS model training component using Unsloth LoRA.
"""

import logging
import sys
from pathlib import Path

from .config import PipelineConfig


class TTSTrainer:
    """Fine-tune a Text-to-Speech model using LoRA with Unsloth."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.lora_dir = config.out_dir / "tts_lora"

    def train(self, dataset_repo: str) -> Path:
        """
        Fine-tune the TTS model on the prepared dataset.
        
        Args:
            dataset_repo: Hugging Face repository containing the training dataset
            
        Returns:
            Path to the saved LoRA adapter
        """
        try:
            from unsloth import FastModel, is_bfloat16_supported
            from transformers import Trainer, TrainingArguments
            from datasets import load_dataset
        except ImportError as e:
            self.logger.error(f"Required packages not installed: {e}")
            self.logger.error("Please install: pip install unsloth[tts] transformers datasets")
            sys.exit(1)

        self.logger.info(f"Loading base model: {self.config.base_tts}")
        
        try:
            model, tokenizer = FastModel.from_pretrained(
                self.config.base_tts, 
                load_in_4bit=False
            )
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            sys.exit(1)

        self.logger.info(f"Loading dataset: {dataset_repo}")
        
        try:
            dataset = load_dataset(dataset_repo, split="train")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        self.logger.info("Setting up training configuration")
        
        training_args = TrainingArguments(
            output_dir=str(self.lora_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=25,
            save_steps=500,
            eval_steps=500,
            warmup_steps=100,
            push_to_hub=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        self.logger.info(f"Starting training for {self.config.max_steps} steps")
        
        try:
            trainer.train()
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            sys.exit(1)

        # Save the LoRA adapter
        self.logger.info(f"Saving LoRA adapter to: {self.lora_dir}")
        model.save_pretrained(str(self.lora_dir))
        
        return self.lora_dir
