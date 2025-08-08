#!/usr/bin/env python3
"""
Simple Whisper Trainer
Combines the simple end-to-end Whisper approach with training capabilities.
Uses only Whisper's built-in feature extraction and processing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import whisper
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleWhisperDataset(Dataset):
    """
    Simplified Dataset class that uses Whisper's built-in processing.
    No external feature extraction required.
    """
    
    def __init__(self, manifest_path: str, model: whisper.Whisper, language: str = "af", max_length: int = 448):
        self.manifest_path = manifest_path
        self.model = model
        self.language = language
        self.max_length = max_length
        self.data = self._load_manifest()
        
        # Get tokenizer from model
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True,
            language=language
        )
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest file"""
        data = []
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        audio_path = item['audio_filepath']
        # Adjust path for NEW SCRIPT directory
        if not audio_path.startswith('../'):
            audio_path = '../' + audio_path
        text = item['text']
        
        try:
            # Use Whisper's built-in audio loading and mel spectrogram extraction
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Extract mel spectrogram using Whisper's built-in method
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Tokenize text using Whisper's tokenizer
            tokens = self.tokenizer.encode(text)
            
            # Pad or truncate tokens
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [self.tokenizer.eot] * (self.max_length - len(tokens))
            
            return {
                'mel_spectrogram': mel,
                'tokens': torch.tensor(tokens, dtype=torch.long),
                'text': text,
                'audio_path': audio_path,
                'success': True
            }
            
        except Exception as e:
            # Return dummy data for failed samples
            dummy_mel = torch.zeros((80, 3000), device=self.model.device)
            dummy_tokens = torch.zeros(self.max_length, dtype=torch.long)
            
            return {
                'mel_spectrogram': dummy_mel,
                'tokens': dummy_tokens,
                'text': "",
                'audio_path': audio_path,
                'success': False,
                'error': str(e)
            }

class SimpleDataCollator:
    """
    Simple data collator for batching Whisper data.
    """
    
    def __call__(self, batch: List[Dict]) -> Dict:
        # Filter out failed samples
        valid_batch = [item for item in batch if item['success']]
        
        if not valid_batch:
            # Return dummy batch if all samples failed
            return {
                'mel_spectrograms': torch.zeros((1, 80, 3000)),
                'tokens': torch.zeros((1, 448), dtype=torch.long),
                'texts': [""],
                'audio_paths': [""],
                'batch_size': 0
            }
        
        # Stack mel spectrograms
        mel_spectrograms = torch.stack([item['mel_spectrogram'] for item in valid_batch])
        
        # Stack tokens
        tokens = torch.stack([item['tokens'] for item in valid_batch])
        
        # Collect texts and paths
        texts = [item['text'] for item in valid_batch]
        audio_paths = [item['audio_path'] for item in valid_batch]
        
        return {
            'mel_spectrograms': mel_spectrograms,
            'tokens': tokens,
            'texts': texts,
            'audio_paths': audio_paths,
            'batch_size': len(valid_batch)
        }

class SimpleWhisperTrainer:
    """
    Simplified Whisper trainer that uses built-in Whisper capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "tiny",
                 language: str = "af",
                 device: Optional[str] = None,
                 learning_rate: float = 1e-5,
                 warmup_steps: int = 100):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        print(f"Initializing Simple Whisper Trainer on {self.device}")
        print(f"Model: {model_name}")
        print(f"Language: {language}")
        
        # Load Whisper model
        self.model = whisper.load_model(model_name, device=self.device)
        self.model_name = model_name
        
        # Setup optimizer (only train decoder parameters)
        decoder_params = list(self.model.decoder.parameters())
        self.optimizer = optim.AdamW(decoder_params, lr=learning_rate)
        
        print(f"Training {len(decoder_params)} decoder parameters")
        
        # Data collator
        self.data_collator = SimpleDataCollator()
        
        # Training history
        self.training_history = {
            'losses': [],
            'learning_rates': [],
            'steps': []
        }
    
    def setup_data_loaders(self, train_manifest: str, val_manifest: str, batch_size: int = 4):
        """Setup training and validation data loaders"""
        
        # Create datasets
        self.train_dataset = SimpleWhisperDataset(
            train_manifest, self.model, self.language
        )
        self.val_dataset = SimpleWhisperDataset(
            val_manifest, self.model, self.language
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0
        )
        
        print(f"Training dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
    
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute training loss"""
        if batch['batch_size'] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        mel_spectrograms = batch['mel_spectrograms']
        tokens = batch['tokens']
        
        # Forward pass through encoder
        audio_features = self.model.encoder(mel_spectrograms)
        
        # Forward pass through decoder
        logits = self.model.decoder(tokens[:, :-1], audio_features)
        
        # Compute cross-entropy loss
        targets = tokens[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0  # Ignore padding tokens
        )
        
        return loss
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch['batch_size'] > 0:
                    loss = self.compute_loss(batch)
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def update_learning_rate(self, step: int):
        """Update learning rate with warmup"""
        if step < self.warmup_steps:
            lr = self.learning_rate * (step + 1) / self.warmup_steps
        else:
            lr = self.learning_rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def save_checkpoint(self, step: int, loss: float, checkpoint_dir: str = "whisper_checkpoints"):
        """Save model checkpoint"""
        # Add this near the beginning of the train method (around line 330):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_name': self.model_name,
            'language': self.language,
            'training_history': self.training_history
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the step number"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        step = checkpoint['step']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded successfully!")
        print(f"Resuming from step: {step}")
        print(f"Previous loss: {loss:.4f}")
        
        return step
    
    def train(self, 
              train_manifest: str,
              val_manifest: str,
              max_steps: int = 1000,
              batch_size: int = 4,
              eval_steps: int = 100,
              save_steps: int = 250,
              checkpoint_dir: str = "whisper_checkpoints",
              start_step: int = 0):
        """Main training loop"""
        
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Max steps: {max_steps}")
        print(f"Batch size: {batch_size}")
        if start_step > 0:
            print(f"Resuming from step: {start_step}")
        
        # Setup data loaders
        self.setup_data_loaders(train_manifest, val_manifest, batch_size)
        
        print("Starting training loop...")
        
        step = start_step
        best_val_loss = float('inf')
        
        while step < max_steps:
            for batch in self.train_loader:
                if step >= max_steps:
                    break
                
                # Update learning rate
                current_lr = self.update_learning_rate(step)
                
                # Training step
                loss = self.train_step(batch)
                
                # Record training history
                self.training_history['losses'].append(loss)
                self.training_history['learning_rates'].append(current_lr)
                self.training_history['steps'].append(step)
                
                # Print progress
                if step % 10 == 0:
                    percentage = (step / max_steps) * 100
                    print(f"Step {step}/{max_steps} ({percentage:.1f}%), Loss: {loss:.4f}, LR: {current_lr:.2e}")
                
                # Evaluation
                if step % eval_steps == 0 and step > 0:
                    val_loss = self.evaluate()
                    percentage = (step / max_steps) * 100
                    print(f"Step {step}/{max_steps} ({percentage:.1f}%), Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = {
                            'step': step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': val_loss,
                            'model_name': self.model_name,
                            'language': self.language,
                            'training_history': self.training_history
                        }
                        torch.save(best_checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
                        print(f"New best model saved with validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                if step % save_steps == 0 and step > 0:
                    self.save_checkpoint(step, loss, checkpoint_dir)
                
                step += 1
        
        print(f"Training completed! Total steps: {step}")
        
        # Final evaluation
        final_val_loss = self.evaluate()
        print(f"Final validation loss: {final_val_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(step, final_val_loss, checkpoint_dir)
        
        # Save training history plot
        self.plot_training_history(checkpoint_dir)
        
        return self.training_history
    
    def plot_training_history(self, save_dir: str):
        """Plot and save training history"""
        if not self.training_history['losses']:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['steps'], self.training_history['losses'])
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['steps'], self.training_history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved: {plot_path}")
    
    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict:
        """Transcribe audio using the trained model"""
        try:
            result = self.model.transcribe(audio_path, language=self.language, **kwargs)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", self.language),
                "segments": result.get("segments", []),
                "success": True,
                "audio_path": audio_path
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False,
                "audio_path": audio_path
            }

def main():
    """
    Main training function - Resume from checkpoint-4500.pt
    """
    print("=" * 60)
    print("Simple Whisper Tiny Model Trainer - Resume from Checkpoint")
    print("Uses only Whisper's built-in capabilities")
    print("=" * 60)
    
    # Initialize trainer for tiny model (to match checkpoint-4500.pt architecture)
    trainer = SimpleWhisperTrainer(
        model_name="tiny",  # Changed from "base" to "tiny" to match checkpoint
        language="af",
        learning_rate=1e-5,  # Original tiny model learning rate
        warmup_steps=100     # Original tiny model warmup steps
    )
    
    # Load checkpoint-4500.pt and get the step number
    checkpoint_path = "whisper_checkpoints/checkpoint-4500.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints in whisper_checkpoints/:")
        if os.path.exists("whisper_checkpoints"):
            for f in os.listdir("whisper_checkpoints"):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return
    
    # Load the checkpoint and get starting step
    start_step = trainer.load_checkpoint(checkpoint_path)
    
    print(f"Resuming training from step {start_step} to 6000 steps...")
    print(f"Remaining steps: {6000 - start_step}")
    
    # Training configuration
    train_manifest = "../processed_data/train_manifest.jsonl"
    val_manifest = "../processed_data/val_manifest.jsonl"
    
    if not os.path.exists(train_manifest):
        print(f"Training manifest not found: {train_manifest}")
        return
    
    if not os.path.exists(val_manifest):
        print(f"Validation manifest not found: {val_manifest}")
        return
    
    # Continue training to 6000 steps
    history = trainer.train(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        max_steps=6000,  # Continue until 6000 steps total
        batch_size=4,
        eval_steps=100,
        save_steps=250,
        checkpoint_dir="whisper_checkpoints",  # Keep same directory as original
        start_step=start_step  # Resume from loaded checkpoint step
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Training resumed from step {start_step} and completed at step 6000")
    print("Key advantages of this approach:")
    print("✓ Uses only Whisper's built-in feature extraction")
    print("✓ No external dependencies for audio processing")
    print("✓ Simplified training pipeline")
    print("✓ Direct integration with Whisper's architecture")
    print("✓ Maintains compatibility with Whisper's inference")
    print("✓ Seamlessly resumes from existing checkpoint")
    print("=" * 60)

if __name__ == "__main__":
    main()