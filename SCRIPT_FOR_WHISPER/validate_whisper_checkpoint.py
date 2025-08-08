#!/usr/bin/env python3
"""
General Whisper Checkpoint Validator
Validates any checkpoint using the validation manifest.
Easily configurable for different checkpoints.
"""

import os
import torch
import whisper
import numpy as np
import json
from typing import Dict, List, Tuple
import jiwer
from pathlib import Path
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class WhisperCheckpointValidator:
    """Whisper checkpoint validator with comprehensive metrics"""
    
    def __init__(self, 
                 model_path: str,
                 model_name: str = "tiny",
                 device: str = "auto",
                 language: str = "af"):
        
        self.device = self._get_device(device)
        self.language = language
        self.model_path = model_path
        
        print(f"Loading checkpoint from: {model_path}")
        
        # Load base model
        self.model = whisper.load_model(model_name, device=self.device)
        
        # Load checkpoint weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        self.model.eval()
        
        # Validation results storage
        self.validation_results = {
            'predictions': [],
            'references': [],
            'wer_scores': [],
            'cer_scores': [],
            'audio_paths': [],
            'processing_times': [],
            'audio_durations': []
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def transcribe_audio(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe single audio file"""
        start_time = time.time()
        
        try:
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe"
            )
            
            transcription = result['text'].strip()
            processing_time = time.time() - start_time
            
            return transcription, processing_time
            
        except Exception as e:
            print(f"❌ Error transcribing {audio_path}: {e}")
            return "", time.time() - start_time
    
    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate WER and CER metrics"""
        try:
            # Normalize text
            ref_normalized = reference.lower().strip()
            hyp_normalized = hypothesis.lower().strip()
            
            # Calculate WER
            wer = jiwer.wer(ref_normalized, hyp_normalized) * 100
            
            # Calculate CER
            cer = jiwer.cer(ref_normalized, hyp_normalized) * 100
            
            return {
                'wer': wer,
                'cer': cer,
                'reference_length': len(ref_normalized.split()),
                'hypothesis_length': len(hyp_normalized.split()),
                'reference_chars': len(ref_normalized),
                'hypothesis_chars': len(hyp_normalized)
            }
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {
                'wer': 100.0,
                'cer': 100.0,
                'reference_length': 0,
                'hypothesis_length': 0,
                'reference_chars': 0,
                'hypothesis_chars': 0
            }
    
    def validate_manifest(self, manifest_path: str, output_dir: str = "validation_results") -> Dict:
        """Validate model on validation manifest"""
        print(f"\n🔍 Starting validation on: {manifest_path}")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = [json.loads(line.strip()) for line in f]
        
        print(f"📊 Validating {len(manifest_data)} samples...")
        
        total_wer = 0.0
        total_cer = 0.0
        total_processing_time = 0.0
        valid_samples = 0
        
        # Define project root directory
        project_root = r"C:\Poroject"
        
        # Process each sample
        for i, item in enumerate(manifest_data):
            audio_path = item['audio_filepath']
            reference_text = item['text']
            audio_duration = item.get('duration', 0)
            
            # Fix audio path - resolve relative to project root
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(project_root, audio_path)
            
            # Progress display
            progress = (i + 1) / len(manifest_data) * 100
            print(f"\r📈 Progress: {progress:.1f}% ({i+1}/{len(manifest_data)})", end="", flush=True)
            
            # Check if audio file exists before trying to transcribe
            if not os.path.exists(audio_path):
                print(f"\n⚠️  Audio file not found: {audio_path}")
                continue
            
            # Transcribe audio
            prediction, processing_time = self.transcribe_audio(audio_path)
            
            if prediction:  # Only process if transcription was successful
                # Calculate metrics
                metrics = self.calculate_metrics(reference_text, prediction)
                
                # Store results
                self.validation_results['predictions'].append(prediction)
                self.validation_results['references'].append(reference_text)
                self.validation_results['wer_scores'].append(metrics['wer'])
                self.validation_results['cer_scores'].append(metrics['cer'])
                self.validation_results['audio_paths'].append(audio_path)
                self.validation_results['processing_times'].append(processing_time)
                self.validation_results['audio_durations'].append(audio_duration)
                
                total_wer += metrics['wer']
                total_cer += metrics['cer']
                total_processing_time += processing_time
                valid_samples += 1
        
        print("\n✅ Validation completed!")
        
        # Calculate summary metrics
        if valid_samples > 0:
            avg_wer = total_wer / valid_samples
            avg_cer = total_cer / valid_samples
            avg_processing_time = total_processing_time / valid_samples
            
            validation_summary = {
                'checkpoint_path': self.model_path,
                'total_samples': len(manifest_data),
                'valid_samples': valid_samples,
                'failed_samples': len(manifest_data) - valid_samples,
                'average_wer': avg_wer,
                'average_cer': avg_cer,
                'min_wer': min(self.validation_results['wer_scores']),
                'max_wer': max(self.validation_results['wer_scores']),
                'min_cer': min(self.validation_results['cer_scores']),
                'max_cer': max(self.validation_results['cer_scores']),
                'std_wer': np.std(self.validation_results['wer_scores']),
                'std_cer': np.std(self.validation_results['cer_scores']),
                'average_processing_time': avg_processing_time,
                'total_processing_time': total_processing_time,
                'real_time_factor': avg_processing_time / np.mean(self.validation_results['audio_durations']) if self.validation_results['audio_durations'] else 0
            }
        else:
            validation_summary = {
                'error': 'No successful validations',
                'checkpoint_path': self.model_path,
                'total_samples': len(manifest_data),
                'valid_samples': 0
            }
        
        # Print summary
        self._print_summary(validation_summary)
        
        # Save results
        self._save_results(output_dir, validation_summary)
        
        # Generate plots
        if valid_samples > 0:
            self._generate_plots(output_dir)
        
        return validation_summary
    
    def _print_summary(self, summary: Dict):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Checkpoint: {os.path.basename(summary['checkpoint_path'])}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Valid samples: {summary['valid_samples']}")
        
        if 'average_wer' in summary:
            print(f"Average WER: {summary['average_wer']:.2f}%")
            print(f"Average CER: {summary['average_cer']:.2f}%")
            print(f"WER Range: {summary['min_wer']:.2f}% - {summary['max_wer']:.2f}%")
            print(f"CER Range: {summary['min_cer']:.2f}% - {summary['max_cer']:.2f}%")
            print(f"WER Std Dev: {summary['std_wer']:.2f}%")
            print(f"CER Std Dev: {summary['std_cer']:.2f}%")
            print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
            print(f"Real-time Factor: {summary['real_time_factor']:.2f}x")
        
        print("="*80)
    
    def _save_results(self, output_dir: str, summary: Dict):
        """Save validation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(self.model_path).replace('.pt', '')
        
        # Save detailed results
        detailed_results = {
            'summary': summary,
            'detailed_results': [
                {
                    'audio_path': path,
                    'reference': ref,
                    'prediction': pred,
                    'wer_score': wer,
                    'cer_score': cer,
                    'processing_time': time,
                    'audio_duration': duration
                }
                for path, ref, pred, wer, cer, time, duration in zip(
                    self.validation_results['audio_paths'],
                    self.validation_results['references'],
                    self.validation_results['predictions'],
                    self.validation_results['wer_scores'],
                    self.validation_results['cer_scores'],
                    self.validation_results['processing_times'],
                    self.validation_results['audio_durations']
                )
            ]
        }
        
        # Save JSON results
        results_path = os.path.join(output_dir, f'validation_results_{checkpoint_name}_{timestamp}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Detailed results saved to: {results_path}")
        
        # Save CSV summary
        if self.validation_results['audio_paths']:
            df = pd.DataFrame({
                'audio_path': [os.path.basename(p) for p in self.validation_results['audio_paths']],
                'reference': self.validation_results['references'],
                'prediction': self.validation_results['predictions'],
                'wer_score': self.validation_results['wer_scores'],
                'cer_score': self.validation_results['cer_scores'],
                'processing_time': self.validation_results['processing_times'],
                'audio_duration': self.validation_results['audio_durations']
            })
            
            csv_path = os.path.join(output_dir, f'validation_summary_{checkpoint_name}_{timestamp}.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"📊 Summary CSV saved to: {csv_path}")
    
    def _generate_plots(self, output_dir: str):
        """Generate validation visualization plots"""
        if not self.validation_results['wer_scores']:
            print("No results to plot")
            return
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # WER distribution histogram
        ax1.hist(self.validation_results['wer_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('WER (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('WER Score Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(self.validation_results['wer_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(self.validation_results["wer_scores"]):.2f}%')
        ax1.legend()
        
        # CER distribution histogram
        ax2.hist(self.validation_results['cer_scores'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('CER (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CER Score Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(self.validation_results['cer_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(self.validation_results["cer_scores"]):.2f}%')
        ax2.legend()
        
        # WER vs CER scatter plot
        ax3.scatter(self.validation_results['wer_scores'], self.validation_results['cer_scores'], alpha=0.6, color='purple')
        ax3.set_xlabel('WER (%)')
        ax3.set_ylabel('CER (%)')
        ax3.set_title('WER vs CER Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Processing time vs WER
        ax4.scatter(self.validation_results['processing_times'], self.validation_results['wer_scores'], alpha=0.6, color='orange')
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('WER (%)')
        ax4.set_title('Processing Time vs WER')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(self.model_path).replace('.pt', '')
        plot_path = os.path.join(output_dir, f'validation_analysis_{checkpoint_name}_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Validation plots saved to: {plot_path}")
    
    def _reset_results(self):
        """Reset validation results for new validation run"""
        self.validation_results = {
            'predictions': [],
            'references': [],
            'wer_scores': [],
            'cer_scores': [],
            'audio_paths': [],
            'processing_times': [],
            'audio_durations': []
        }

if __name__ == "__main__":
    print("="*80)
    print("WHISPER CHECKPOINT VALIDATION - CHECKPOINT-4500.PT")
    print("="*80)
    
    # Updated checkpoint path to checkpoint-4500.pt
    checkpoint_path = r"C:\Poroject\NEW SCRIPT\whisper_checkpoints\checkpoint-4500.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoints_dir = r"C:\Poroject\NEW SCRIPT\whisper_checkpoints"
        if os.path.exists(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        exit(1)
    
    # Validation manifest path
    val_manifest = r"C:\Poroject\processed_data\val_manifest.jsonl"
    
    if not os.path.exists(val_manifest):
        print(f"❌ Error: Validation manifest not found at {val_manifest}")
        exit(1)
    
    print(f"🔍 Validating checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"📋 Using validation manifest: {val_manifest}")
    
    try:
        # Initialize validator with checkpoint-4500.pt
        validator = WhisperCheckpointValidator(
            model_path=checkpoint_path,
            model_name="tiny",  # Keep as tiny since checkpoint-4500.pt is from tiny model
            device="auto",
            language="af"
        )
        
        # Run validation
        results = validator.validate_manifest(
            manifest_path=val_manifest,
            output_dir=r"C:\Poroject\NEW SCRIPT\validation_results"
        )

        print("\n🎉 Validation completed successfully!")
        print(f"📊 Results saved to: C:/Poroject/NEW SCRIPT/validation_results")

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)