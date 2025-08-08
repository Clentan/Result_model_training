#!/usr/bin/env python3
"""
Simple Checkpoint-980 Tester with Individual WER Display
Tests checkpoint-980.pt using only Whisper's built-in processing.
Shows WER for each audio file and progress percentage.
"""

import os
import torch
import whisper
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import jiwer
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SimpleWhisperTester:
    """Simple testing suite for Whisper models using built-in processing only"""
    
    def __init__(self, 
                 model_path: str = None,
                 model_name: str = "tiny",
                 device: str = "auto",
                 language: str = "af"):
        
        self.device = self._get_device(device)
        self.language = language
        self.model_name = model_name
        self.model_path = model_path
        
        # Load model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = whisper.load_model(model_name, device=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_finetuned = True
        else:
            print(f"Loading pre-trained model: {model_name}")
            self.model = whisper.load_model(model_name, device=self.device)
            self.is_finetuned = False
        
        self.model.eval()
        
        # Test results storage
        self.test_results = {
            'predictions': [],
            'references': [],
            'wer_scores': [],
            'cer_scores': [],
            'audio_paths': [],
            'processing_times': [],
            'audio_durations': [],
            'word_counts': [],
            'char_counts': []
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def transcribe_audio(self, audio_path: str) -> Tuple[str, float, float]:
        """Transcribe single audio file with timing using Whisper's built-in processing"""
        start_time = time.time()
        
        # Handle relative paths - convert to absolute path from project root
        if not os.path.isabs(audio_path):
            # Get the project root directory (parent of NEW SCRIPT)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            audio_path = os.path.join(project_root, audio_path)
        
        try:
            # Use Whisper's built-in audio loading
            audio = whisper.load_audio(audio_path)
            duration = len(audio) / whisper.audio.SAMPLE_RATE
            
            # Use Whisper's built-in transcription with optimized parameters
            result = whisper.transcribe(
                self.model, 
                audio_path, 
                language=self.language,
                task="transcribe",
                verbose=False,
                temperature=0.0,  # Use deterministic decoding for consistency
                beam_size=5,      # Use beam search for better accuracy
                best_of=5,        # Generate multiple candidates and pick best
                patience=1.0,     # Patience for beam search
            )
            
            processing_time = time.time() - start_time
            transcription = result['text'].strip()
            
            return transcription, processing_time, duration
            
        except Exception as e:
            print(f"\nError transcribing {audio_path}: {e}")
            return "", 0.0, 0.0
    
    def calculate_metrics(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """Calculate WER and CER metrics"""
        try:
            # WER calculation
            wer = jiwer.wer(reference, hypothesis) * 100
            
            # CER calculation
            cer = jiwer.cer(reference, hypothesis) * 100
            
            return wer, cer
        except:
            return 100.0, 100.0  # Return max error if calculation fails
    
    def test_manifest(self, manifest_path: str, output_dir: str = "test_results") -> Dict:
        """Test model on manifest file with detailed individual results"""
        print(f"\n🔍 Loading test manifest: {manifest_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load manifest
        test_samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
        
        total_samples = len(test_samples)
        print(f"📊 Found {total_samples} test samples")
        print(f"\n{'='*80}")
        print(f"TESTING CHECKPOINT-1750.PT - INDIVIDUAL RESULTS")
        print(f"{'='*80}")
        
        # Reset results
        self._reset_results()
        
        # Process each sample
        valid_samples = 0
        total_wer = 0
        total_cer = 0
        total_processing_time = 0
        total_audio_duration = 0
        
        for i, sample in enumerate(test_samples):
            # Calculate and display progress
            progress_percent = ((i + 1) / total_samples) * 100
            remaining_samples = total_samples - (i + 1)
            
            print(f"\n[{i+1:3d}/{total_samples}] ({progress_percent:5.1f}%) - {remaining_samples} tests remaining")
            
            audio_path = sample['audio_filepath']
            reference = sample['text'].strip()
            
            # Display audio file info
            audio_filename = os.path.basename(audio_path)
            print(f"🎵 Audio: {audio_filename}")
            print(f"📝 Reference: {reference[:80]}{'...' if len(reference) > 80 else ''}")
            
            # Transcribe audio
            start_time = time.time()
            prediction, proc_time, duration = self.transcribe_audio(audio_path)
            
            if prediction:  # Only process if transcription succeeded
                # Calculate metrics
                wer, cer = self.calculate_metrics(reference, prediction)
                
                # Display results for this audio
                print(f"🤖 Predicted: {prediction[:80]}{'...' if len(prediction) > 80 else ''}")
                print(f"📊 WER: {wer:6.2f}% | CER: {cer:6.2f}% | Time: {proc_time:5.2f}s | Duration: {duration:5.2f}s")
                
                # Color coding for WER
                if wer <= 10:
                    status = "✅ EXCELLENT"
                elif wer <= 25:
                    status = "🟢 GOOD"
                elif wer <= 50:
                    status = "🟡 FAIR"
                else:
                    status = "🔴 POOR"
                
                print(f"🎯 Status: {status}")
                
                # Store results
                self.test_results['predictions'].append(prediction)
                self.test_results['references'].append(reference)
                self.test_results['wer_scores'].append(wer)
                self.test_results['cer_scores'].append(cer)
                self.test_results['audio_paths'].append(audio_path)
                self.test_results['processing_times'].append(proc_time)
                self.test_results['audio_durations'].append(duration)
                self.test_results['word_counts'].append(len(reference.split()))
                self.test_results['char_counts'].append(len(reference))
                
                # Accumulate totals
                total_wer += wer
                total_cer += cer
                total_processing_time += proc_time
                total_audio_duration += duration
                valid_samples += 1
                
                # Show running average
                running_avg_wer = total_wer / valid_samples
                print(f"📈 Running Avg WER: {running_avg_wer:6.2f}%")
                
            else:
                print(f"❌ FAILED to transcribe")
            
            print(f"{'-'*80}")
        
        print(f"\n✅ Completed testing {valid_samples} samples")
        
        # Calculate comprehensive metrics
        summary = self._calculate_comprehensive_metrics(
            valid_samples, total_wer, total_cer, total_processing_time, total_audio_duration
        )
        
        # Print final summary
        self._print_final_summary(summary)
        
        # Save detailed results
        self._save_detailed_results(output_dir, summary)
        
        # Generate plots
        self._generate_comprehensive_plots(output_dir)
        
        return summary
    
    def _calculate_comprehensive_metrics(self, valid_samples, total_wer, total_cer, total_processing_time, total_audio_duration):
        """Calculate comprehensive metrics"""
        if valid_samples == 0:
            return {'error': 'No valid samples processed'}
        
        avg_wer = total_wer / valid_samples
        avg_cer = total_cer / valid_samples
        real_time_factor = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
        
        # Calculate WER distribution
        excellent_count = sum(1 for wer in self.test_results['wer_scores'] if wer <= 10)
        good_count = sum(1 for wer in self.test_results['wer_scores'] if 10 < wer <= 25)
        fair_count = sum(1 for wer in self.test_results['wer_scores'] if 25 < wer <= 50)
        poor_count = sum(1 for wer in self.test_results['wer_scores'] if wer > 50)
        
        return {
            'model_info': {
                'model_path': self.model_path,
                'model_name': self.model_name,
                'is_finetuned': self.is_finetuned,
                'device': self.device,
                'language': self.language
            },
            'performance_metrics': {
                'average_wer': avg_wer,
                'average_cer': avg_cer,
                'accuracy': 100 - avg_wer,
                'total_samples': valid_samples,
                'total_processing_time': total_processing_time,
                'total_audio_duration': total_audio_duration,
                'real_time_factor': real_time_factor
            },
            'wer_distribution': {
                'excellent_count': excellent_count,
                'good_count': good_count,
                'fair_count': fair_count,
                'poor_count': poor_count,
                'excellent_percent': (excellent_count / valid_samples) * 100,
                'good_percent': (good_count / valid_samples) * 100,
                'fair_percent': (fair_count / valid_samples) * 100,
                'poor_percent': (poor_count / valid_samples) * 100
            },
            'detailed_results': {
                'predictions': self.test_results['predictions'],
                'references': self.test_results['references'],
                'wer_scores': self.test_results['wer_scores'],
                'cer_scores': self.test_results['cer_scores'],
                'audio_paths': self.test_results['audio_paths'],
                'processing_times': self.test_results['processing_times'],
                'audio_durations': self.test_results['audio_durations']
            }
        }
    
    def _print_final_summary(self, summary):
        """Print comprehensive final summary"""
        if 'error' in summary:
            print(f"❌ Error: {summary['error']}")
            return
        
        metrics = summary['performance_metrics']
        model_info = summary['model_info']
        distribution = summary['wer_distribution']
        
        print(f"\n{'='*80}")
        print(f"🎉 CHECKPOINT-2000 FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"📋 Model Info:")
        print(f"   Model: {model_info['model_name']} ({'Fine-tuned' if model_info['is_finetuned'] else 'Pre-trained'})")
        print(f"   Checkpoint: checkpoint-2000.pt")
        print(f"   Device: {model_info['device']}")
        print(f"   Language: {model_info['language']}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Average WER: {metrics['average_wer']:.2f}%")
        print(f"   Average CER: {metrics['average_cer']:.2f}%")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Total Samples: {metrics['total_samples']}")
        print(f"   Processing Time: {metrics['total_processing_time']:.2f}s")
        print(f"   Real-time Factor: {metrics['real_time_factor']:.2f}x")
        
        print(f"\n🎯 WER Distribution:")
        print(f"   ✅ Excellent (≤10%):  {distribution['excellent_count']:3d} samples ({distribution['excellent_percent']:5.1f}%)")
        print(f"   🟢 Good (10-25%):     {distribution['good_count']:3d} samples ({distribution['good_percent']:5.1f}%)")
        print(f"   🟡 Fair (25-50%):     {distribution['fair_count']:3d} samples ({distribution['fair_percent']:5.1f}%)")
        print(f"   🔴 Poor (>50%):       {distribution['poor_count']:3d} samples ({distribution['poor_percent']:5.1f}%)")
        
        # Best and worst performing samples
        if self.test_results['wer_scores']:
            best_idx = np.argmin(self.test_results['wer_scores'])
            worst_idx = np.argmax(self.test_results['wer_scores'])
            
            print(f"\n🏆 Best Performance:")
            print(f"   File: {os.path.basename(self.test_results['audio_paths'][best_idx])}")
            print(f"   WER: {self.test_results['wer_scores'][best_idx]:.2f}%")
            
            print(f"\n⚠️  Worst Performance:")
            print(f"   File: {os.path.basename(self.test_results['audio_paths'][worst_idx])}")
            print(f"   WER: {self.test_results['wer_scores'][worst_idx]:.2f}%")
        
        print(f"{'='*80}")
    
    def _save_detailed_results(self, output_dir, summary):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = os.path.join(output_dir, f"checkpoint_980_detailed_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save CSV with individual results
        csv_file = os.path.join(output_dir, f"checkpoint_980_individual_results_{timestamp}.csv")
        df_data = {
            'audio_file': [os.path.basename(path) for path in self.test_results['audio_paths']],
            'wer_score': self.test_results['wer_scores'],
            'cer_score': self.test_results['cer_scores'],
            'processing_time': self.test_results['processing_times'],
            'audio_duration': self.test_results['audio_durations'],
            'reference': self.test_results['references'],
            'prediction': self.test_results['predictions']
        }
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"\n💾 Results saved to:")
        print(f"   JSON: {results_file}")
        print(f"   CSV:  {csv_file}")
    
    def _generate_comprehensive_plots(self, output_dir):
        """Generate comprehensive analysis plots"""
        if not self.test_results['wer_scores']:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Checkpoint-2000 Detailed Test Analysis', fontsize=16, fontweight='bold')
        
        # WER Distribution
        axes[0, 0].hist(self.test_results['wer_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('WER Score Distribution')
        axes[0, 0].set_xlabel('WER (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(self.test_results['wer_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(self.test_results["wer_scores"]):.1f}%')
        axes[0, 0].legend()
        
        # Processing Time vs Audio Duration
        axes[0, 1].scatter(self.test_results['audio_durations'], self.test_results['processing_times'], alpha=0.6)
        axes[0, 1].set_title('Processing Time vs Audio Duration')
        axes[0, 1].set_xlabel('Audio Duration (s)')
        axes[0, 1].set_ylabel('Processing Time (s)')
        axes[0, 1].plot([0, max(self.test_results['audio_durations'])], [0, max(self.test_results['audio_durations'])], 'r--', label='Real-time line')
        axes[0, 1].legend()
        
        # WER vs Word Count
        axes[1, 0].scatter(self.test_results['word_counts'], self.test_results['wer_scores'], alpha=0.6)
        axes[1, 0].set_title('WER vs Word Count')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('WER (%)')
        
        # CER vs WER
        axes[1, 1].scatter(self.test_results['wer_scores'], self.test_results['cer_scores'], alpha=0.6)
        axes[1, 1].set_title('CER vs WER')
        axes[1, 1].set_xlabel('WER (%)')
        axes[1, 1].set_ylabel('CER (%)')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, f"checkpoint_2000_detailed_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Analysis plots saved to: {plot_file}")
    
    def _reset_results(self):
        """Reset test results for new test run"""
        self.test_results = {
            'predictions': [],
            'references': [],
            'wer_scores': [],
            'cer_scores': [],
            'audio_paths': [],
            'processing_times': [],
            'audio_durations': [],
            'word_counts': [],
            'char_counts': []
        }

def create_comprehensive_visualization(results: Dict, output_dir: str, checkpoint_name: str):
    """Create comprehensive 3x3 matplotlib visualization of test results"""
    
    # Extract data for plotting
    detailed_results = results['detailed_results']
    wer_scores = detailed_results['wer_scores']
    cer_scores = detailed_results['cer_scores']
    processing_times = detailed_results['processing_times']
    audio_durations = detailed_results['audio_durations']
    predictions = detailed_results['predictions']
    references = detailed_results['references']
    
    # Extract training history if available
    training_history = results['model_info'].get('training_history')
    
    # Calculate additional metrics
    word_counts = [len(ref.split()) for ref in references]
    char_counts = [len(ref) for ref in references]
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f'Comprehensive Whisper Model Analysis - Checkpoint {checkpoint_name}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. WER Score Distribution (Top Left)
    ax1 = axes[0, 0]
    ax1.hist(wer_scores, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(np.mean(wer_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(wer_scores):.1f}%')
    ax1.set_title('WER Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('WER (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CER Score Distribution (Top Center)
    ax2 = axes[0, 1]
    ax2.hist(cer_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(np.mean(cer_scores), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cer_scores):.1f}%')
    ax2.set_title('CER Score Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('CER (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. WER vs CER Correlation (Top Right)
    ax3 = axes[0, 2]
    ax3.scatter(wer_scores, cer_scores, alpha=0.6, color='red', s=20)
    z = np.polyfit(wer_scores, cer_scores, 1)
    p = np.poly1d(z)
    ax3.plot(wer_scores, p(wer_scores), "b--", alpha=0.8, linewidth=2)
    correlation = np.corrcoef(wer_scores, cer_scores)[0, 1]
    ax3.set_title('WER vs CER Correlation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('WER (%)')
    ax3.set_ylabel('CER (%)')
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    
    # 4. Processing Time vs Audio Duration (Middle Left)
    ax4 = axes[1, 0]
    ax4.scatter(audio_durations, processing_times, alpha=0.6, color='purple', s=20)
    z = np.polyfit(audio_durations, processing_times, 1)
    p = np.poly1d(z)
    ax4.plot(audio_durations, p(audio_durations), "r--", alpha=0.8, linewidth=2)
    ax4.set_title('Processing Time vs Audio Duration', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Audio Duration (s)')
    ax4.set_ylabel('Processing Time (s)')
    rtf = np.mean(processing_times) / np.mean(audio_durations) if np.mean(audio_durations) > 0 else 0
    ax4.text(0.05, 0.95, f'Real-time Factor: {rtf:.3f}x', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax4.grid(True, alpha=0.3)
    
    # 5. WER vs Sentence Length (Middle Center)
    ax5 = axes[1, 1]
    ax5.scatter(word_counts, wer_scores, alpha=0.6, color='orange', s=20)
    z = np.polyfit(word_counts, wer_scores, 1)
    p = np.poly1d(z)
    ax5.plot(word_counts, p(word_counts), "g--", alpha=0.8, linewidth=2)
    ax5.set_title('WER vs Sentence Length', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Word Count')
    ax5.set_ylabel('WER (%)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Types Distribution (Middle Right)
    ax6 = axes[1, 2]
    # Calculate error types based on WER ranges
    substitutions = sum(1 for wer in wer_scores if 10 < wer <= 50)
    deletions = sum(1 for wer in wer_scores if wer <= 10)
    insertions = sum(1 for wer in wer_scores if wer > 50)
    
    error_types = ['Substitutions', 'Deletions', 'Insertions']
    error_counts = [substitutions, deletions, insertions]
    colors = ['red', 'orange', 'yellow']
    
    bars = ax6.bar(error_types, error_counts, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_title('Error Types Distribution', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, error_counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 7. WER Across Test Samples (Bottom Left)
    ax7 = axes[2, 0]
    sample_indices = range(len(wer_scores))
    ax7.plot(sample_indices, wer_scores, color='blue', alpha=0.7, linewidth=1)
    ax7.axhline(np.mean(wer_scores), color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax7.set_title('WER Across Test Samples', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('WER (%)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Real-time Factor Distribution (Bottom Center)
    ax8 = axes[2, 1]
    rtf_values = [pt/ad if ad > 0 else 0 for pt, ad in zip(processing_times, audio_durations)]
    ax8.hist(rtf_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax8.axvline(np.mean(rtf_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rtf_values):.3f}x')
    ax8.set_title('Real-time Factor Distribution', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Real-time Factor')
    ax8.set_ylabel('Frequency')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Training Loss History (Bottom Left) - NEW
    ax8 = axes[2, 0]
    if training_history and 'losses' in training_history:
        steps = training_history['steps']
        losses = training_history['losses']
        ax8.plot(steps, losses, color='red', linewidth=2)
        ax8.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Training Steps')
        ax8.set_ylabel('Loss')
        ax8.grid(True, alpha=0.3)
        
        # Add average loss annotation
        avg_loss = np.mean(losses)
        ax8.axhline(avg_loss, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Avg Loss: {avg_loss:.4f}')
        ax8.legend()
    else:
        ax8.text(0.5, 0.5, 'No Training\nHistory Available', 
                ha='center', va='center', fontsize=12, transform=ax8.transAxes)
        ax8.set_title('Training Loss History', fontsize=14, fontweight='bold')
    
    # 9. Learning Rate History (Bottom Center) - NEW
    ax9 = axes[2, 1]
    if training_history and 'learning_rates' in training_history:
        steps = training_history['steps']
        learning_rates = training_history['learning_rates']
        ax9.plot(steps, learning_rates, color='green', linewidth=2)
        ax9.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Training Steps')
        ax9.set_ylabel('Learning Rate')
        ax9.grid(True, alpha=0.3)
        ax9.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Add average LR annotation
        avg_lr = np.mean(learning_rates)
        ax9.axhline(avg_lr, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Avg LR: {avg_lr:.2e}')
        ax9.legend()
    else:
        ax9.text(0.5, 0.5, 'No Learning Rate\nHistory Available', 
                ha='center', va='center', fontsize=12, transform=ax9.transAxes)
        ax9.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # 10. Training Summary Box (Bottom Right) - UPDATED
    ax10 = axes[2, 2]
    ax10.axis('off')
    
    # Build summary text with training info
    summary_text = f"""TEST SUMMARY
    
Samples: {len(wer_scores)}
Avg WER: {np.mean(wer_scores):.1f}%
Avg CER: {np.mean(cer_scores):.1f}%
Std WER: {np.std(wer_scores):.1f}%

Model: {results['model_info']['model_name']}
Type: {'Fine-tuned' if results['model_info']['is_finetuned'] else 'Pre-trained'}
Device: {results['model_info']['device']}

Processing:
Total Time: {sum(processing_times):.1f}s
Avg Time/Sample: {np.mean(processing_times):.2f}s"""
    
    # Add training summary if available
    if training_history and 'losses' in training_history:
        final_loss = training_history['losses'][-1] if training_history['losses'] else 'N/A'
        avg_loss = np.mean(training_history['losses']) if training_history['losses'] else 'N/A'
        total_steps = len(training_history['losses']) if training_history['losses'] else 'N/A'
        
        training_summary = f"""

TRAINING SUMMARY:
Total Steps: {total_steps}
Final Loss: {final_loss:.4f if isinstance(final_loss, (int, float)) else final_loss}
Avg Loss: {avg_loss:.4f if isinstance(avg_loss, (int, float)) else avg_loss}"""
        summary_text += training_summary
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'checkpoint_{checkpoint_name}_detailed_analysis_{timestamp}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive analysis plots saved to: {plot_path}")
    plt.show()
    
    return plot_path

if __name__ == "__main__":
    pass  # Script execution code is already above
    print("="*80)
    print("TESTING CHECKPOINT-5000.PT MODEL (DETAILED VERSION)")
    print("="*80)
    
    # Checkpoint path - updated to checkpoint-5000.pt from whisper_checkpoints directory
    checkpoint_path = r"C:\Poroject\NEW SCRIPT\whisper_checkpoints\checkpoint-5000.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure checkpoint-5000.pt exists in the NEW SCRIPT/whisper_checkpoints directory.")
        exit(1)
    
    print(f"Testing checkpoint-5000 with comprehensive analysis...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialize tester with checkpoint-5000.pt
        tester = SimpleWhisperTester(
            model_path=checkpoint_path,  # Use checkpoint-5000.pt
            model_name="tiny",  # Keep as "tiny" since checkpoint-5000.pt is from tiny model training
            device="auto",
            language="af"
        )
        
        # Use test manifest instead of directory-based testing
        test_manifest = r"C:\Poroject\processed_data\test_manifest.jsonl"
        
        if not os.path.exists(test_manifest):
            print(f"❌ Error: Test manifest file not found at {test_manifest}")
            exit(1)
        
        # Run test using manifest (correct method)
        results = tester.test_manifest(
            manifest_path=test_manifest,
            output_dir=r"C:\Poroject\NEW SCRIPT\test_results"
        )
        
        # Save results
        output_file = f"checkpoint_5000_test_results_{timestamp}.json"
        
        # Save to NEW SCRIPT test_results directory
        results_dir = r"C:\Poroject\NEW SCRIPT\test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Results saved to: {output_path}")
        
        # Generate visualization
        print("\n🎨 Creating comprehensive matplotlib visualization...")
        plot_path = create_comprehensive_visualization(results, results_dir, "5000")

        print("\n✅ Testing and visualization completed successfully!")

    except Exception as e:
        print(f"❌ Error testing checkpoint-5000: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Checkpoint path - updated to checkpoint-1700.pt
    checkpoint_path = r"C:\Poroject\NEW SCRIPT\whisper_checkpoints\checkpoint-2000.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure checkpoint-1700.pt exists in the NEW SCRIPT/whisper_checkpoints directory.")
        exit(1)
    
    print(f"Testing checkpoint-1700 with comprehensive analysis...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialize tester with checkpoint-1700.pt
        tester = SimpleWhisperTester(
            model_path=checkpoint_path,  # Use checkpoint-1700.pt
            model_name="tiny",
            device="auto",
            language="af"
        )
        
        # Use test manifest instead of directory-based testing
        test_manifest = r"C:\Poroject\processed_data\test_manifest.jsonl"
        
        if not os.path.exists(test_manifest):
            print(f"❌ Error: Test manifest file not found at {test_manifest}")
            exit(1)
        
        # Run test using manifest (correct method)
        results = tester.test_manifest(
            manifest_path=test_manifest,
            output_dir=r"C:\Poroject\NEW SCRIPT\test_results"
        )
        
        # Save results
        output_file = f"checkpoint_1750_test_results_{timestamp}.json"
        
        # Save to NEW SCRIPT test_results directory
        results_dir = r"C:\Poroject\NEW SCRIPT\test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Results saved to: {output_path}")
        
        # Create comprehensive matplotlib visualization
        print("\n🎨 Creating comprehensive matplotlib visualization...")
        plot_path = create_comprehensive_visualization(results, results_dir, "2000")
        
        print("\n✅ Testing and visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing checkpoint-2000: {e}")
        import traceback
        traceback.print_exc()
        exit(1)