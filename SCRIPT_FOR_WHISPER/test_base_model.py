#!/usr/bin/env python3
"""
Base Model Tester
Tests the trained base model checkpoints using Whisper's built-in processing.
Shows WER for each audio file and comprehensive analysis.
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

class BaseModelTester:
    """Testing suite for Base Whisper models using built-in processing only"""
    
    def __init__(self, 
                 model_path: str = None,
                 model_name: str = "base",
                 device: str = "auto",
                 language: str = "af"):
        
        self.device = self._get_device(device)
        self.language = language
        self.model_name = model_name
        self.model_path = model_path
        
        # Load model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned base model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
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
            return result['text'].strip(), processing_time, duration
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            processing_time = time.time() - start_time
            return "", processing_time, 0.0
    
    def calculate_metrics(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """Calculate WER and CER metrics"""
        try:
            # Calculate WER
            wer = jiwer.wer(reference, hypothesis) * 100
            
            # Calculate CER
            cer = jiwer.cer(reference, hypothesis) * 100
            
            return wer, cer
        except:
            return 100.0, 100.0  # Return 100% error if calculation fails
    
    def test_manifest(self, manifest_path: str, output_dir: str = "test_results") -> Dict:
        """Test model on manifest file with comprehensive analysis"""
        print(f"\n🔍 Testing model on: {manifest_path}")
        print(f"📊 Model: {self.model_name} ({'Fine-tuned' if self.is_finetuned else 'Pre-trained'})")
        print(f"🌐 Language: {self.language}")
        print(f"💻 Device: {self.device}")
        
        # Reset results
        self._reset_results()
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = [json.loads(line) for line in f]
        
        total_samples = len(manifest_data)
        print(f"📁 Total samples to test: {total_samples}")
        print("\n" + "="*80)
        
        valid_samples = 0
        total_wer = 0.0
        total_cer = 0.0
        total_processing_time = 0.0
        total_audio_duration = 0.0
        
        for i, sample in enumerate(manifest_data):
            audio_path = sample['audio_filepath']
            reference = sample['text']
            
            # Progress indicator
            progress = (i + 1) / total_samples * 100
            print(f"\r🔄 Processing: {i+1}/{total_samples} ({progress:.1f}%)", end="", flush=True)
            
            # Transcribe audio
            prediction, proc_time, duration = self.transcribe_audio(audio_path)
            
            if prediction:  # Only process if transcription was successful
                # Calculate metrics
                wer, cer = self.calculate_metrics(reference, prediction)
                
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
                
                # Show individual results every 10 samples
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"\n📝 Sample {i+1}: WER={wer:.1f}%, CER={cer:.1f}%")
                    print(f"   Ref: {reference[:60]}{'...' if len(reference) > 60 else ''}")
                    print(f"   Hyp: {prediction[:60]}{'...' if len(prediction) > 60 else ''}")
        
        print("\n" + "="*80)
        
        # Calculate comprehensive metrics
        summary = self._calculate_comprehensive_metrics(
            valid_samples, total_wer, total_cer, total_processing_time, total_audio_duration
        )
        
        # Print final summary
        self._print_final_summary(summary)
        
        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        self._save_detailed_results(output_dir, summary)
        
        # Generate comprehensive plots
        self._generate_comprehensive_plots(output_dir)
        
        return summary
    
    def _calculate_comprehensive_metrics(self, valid_samples, total_wer, total_cer, total_processing_time, total_audio_duration):
        """Calculate comprehensive metrics and statistics"""
        if valid_samples == 0:
            return {'error': 'No valid samples processed'}
        
        # Basic metrics
        avg_wer = total_wer / valid_samples
        avg_cer = total_cer / valid_samples
        
        # Advanced statistics
        wer_scores = np.array(self.test_results['wer_scores'])
        cer_scores = np.array(self.test_results['cer_scores'])
        
        # Performance metrics
        real_time_factor = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
        
        # Error distribution
        perfect_predictions = np.sum(wer_scores == 0)
        good_predictions = np.sum(wer_scores <= 10)
        acceptable_predictions = np.sum(wer_scores <= 25)
        
        return {
            'model_info': {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'is_finetuned': self.is_finetuned,
                'device': self.device,
                'language': self.language
            },
            'test_metrics': {
                'total_samples': len(self.test_results['predictions']),
                'valid_samples': valid_samples,
                'avg_wer': avg_wer,
                'avg_cer': avg_cer,
                'median_wer': np.median(wer_scores),
                'median_cer': np.median(cer_scores),
                'std_wer': np.std(wer_scores),
                'std_cer': np.std(cer_scores),
                'min_wer': np.min(wer_scores),
                'max_wer': np.max(wer_scores),
                'min_cer': np.min(cer_scores),
                'max_cer': np.max(cer_scores)
            },
            'performance_metrics': {
                'total_processing_time': total_processing_time,
                'total_audio_duration': total_audio_duration,
                'avg_processing_time': total_processing_time / valid_samples,
                'real_time_factor': real_time_factor
            },
            'error_distribution': {
                'perfect_predictions': perfect_predictions,
                'good_predictions': good_predictions,
                'acceptable_predictions': acceptable_predictions,
                'perfect_percentage': (perfect_predictions / valid_samples) * 100,
                'good_percentage': (good_predictions / valid_samples) * 100,
                'acceptable_percentage': (acceptable_predictions / valid_samples) * 100
            }
        }
    
    def _print_final_summary(self, summary):
        """Print comprehensive final summary"""
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Model info
        model_info = summary['model_info']
        print(f"🤖 Model: {model_info['model_name']} ({'Fine-tuned' if model_info['is_finetuned'] else 'Pre-trained'})")
        if model_info['model_path']:
            print(f"📁 Checkpoint: {os.path.basename(model_info['model_path'])}")
        print(f"🌐 Language: {model_info['language']}")
        print(f"💻 Device: {model_info['device']}")
        
        # Test metrics
        metrics = summary['test_metrics']
        print(f"\n📈 ACCURACY METRICS:")
        print(f"   • Average WER: {metrics['avg_wer']:.2f}%")
        print(f"   • Average CER: {metrics['avg_cer']:.2f}%")
        print(f"   • Median WER: {metrics['median_wer']:.2f}%")
        print(f"   • WER Range: {metrics['min_wer']:.1f}% - {metrics['max_wer']:.1f}%")
        print(f"   • WER Std Dev: {metrics['std_wer']:.2f}%")
        
        # Performance metrics
        perf = summary['performance_metrics']
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   • Total Audio: {perf['total_audio_duration']:.1f}s")
        print(f"   • Processing Time: {perf['total_processing_time']:.1f}s")
        print(f"   • Real-time Factor: {perf['real_time_factor']:.2f}x")
        print(f"   • Avg Time/Sample: {perf['avg_processing_time']:.2f}s")
        
        # Error distribution
        dist = summary['error_distribution']
        print(f"\n🎯 ERROR DISTRIBUTION:")
        print(f"   • Perfect (0% WER): {dist['perfect_predictions']} ({dist['perfect_percentage']:.1f}%)")
        print(f"   • Good (≤10% WER): {dist['good_predictions']} ({dist['good_percentage']:.1f}%)")
        print(f"   • Acceptable (≤25% WER): {dist['acceptable_predictions']} ({dist['acceptable_percentage']:.1f}%)")
        
        print(f"\n📊 Total Samples Tested: {metrics['valid_samples']}")
        print("="*80)
    
    def _save_detailed_results(self, output_dir, summary):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine checkpoint name for filename
        checkpoint_name = "base_model"
        if self.model_path:
            checkpoint_name = os.path.splitext(os.path.basename(self.model_path))[0]
        
        # Save detailed JSON results (convert numpy types to Python types)
        detailed_results = {
            'summary': self._convert_numpy_types(summary),
            'individual_results': [
                {
                    'audio_path': str(self.test_results['audio_paths'][i]),
                    'reference': str(self.test_results['references'][i]),
                    'prediction': str(self.test_results['predictions'][i]),
                    'wer': float(self.test_results['wer_scores'][i]),
                    'cer': float(self.test_results['cer_scores'][i]),
                    'processing_time': float(self.test_results['processing_times'][i]),
                    'audio_duration': float(self.test_results['audio_durations'][i]),
                    'word_count': int(self.test_results['word_counts'][i]),
                    'char_count': int(self.test_results['char_counts'][i])
                }
                for i in range(len(self.test_results['predictions']))
            ]
        }
        
        json_path = os.path.join(output_dir, f"{checkpoint_name}_detailed_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_data = {
            'Audio_Path': self.test_results['audio_paths'],
            'Reference': self.test_results['references'],
            'Prediction': self.test_results['predictions'],
            'WER': self.test_results['wer_scores'],
            'CER': self.test_results['cer_scores'],
            'Processing_Time': self.test_results['processing_times'],
            'Audio_Duration': self.test_results['audio_durations']
        }
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{checkpoint_name}_individual_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\n💾 Detailed results saved:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")
    
    def _generate_comprehensive_plots(self, output_dir):
        """Generate comprehensive analysis plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine checkpoint name for filename
        checkpoint_name = "base_model"
        if self.model_path:
            checkpoint_name = os.path.splitext(os.path.basename(self.model_path))[0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Base Model Test Analysis - {checkpoint_name}', fontsize=16, fontweight='bold')
        
        # 1. WER Distribution
        axes[0, 0].hist(self.test_results['wer_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('WER Distribution')
        axes[0, 0].set_xlabel('Word Error Rate (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(self.test_results['wer_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(self.test_results["wer_scores"]):.1f}%')
        axes[0, 0].legend()
        
        # 2. CER Distribution
        axes[0, 1].hist(self.test_results['cer_scores'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('CER Distribution')
        axes[0, 1].set_xlabel('Character Error Rate (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(self.test_results['cer_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(self.test_results["cer_scores"]):.1f}%')
        axes[0, 1].legend()
        
        # 3. WER vs Audio Duration
        axes[0, 2].scatter(self.test_results['audio_durations'], self.test_results['wer_scores'], alpha=0.6, color='green')
        axes[0, 2].set_title('WER vs Audio Duration')
        axes[0, 2].set_xlabel('Audio Duration (seconds)')
        axes[0, 2].set_ylabel('Word Error Rate (%)')
        
        # 4. WER vs Word Count
        axes[1, 0].scatter(self.test_results['word_counts'], self.test_results['wer_scores'], alpha=0.6, color='orange')
        axes[1, 0].set_title('WER vs Word Count')
        axes[1, 0].set_xlabel('Reference Word Count')
        axes[1, 0].set_ylabel('Word Error Rate (%)')
        
        # 5. Processing Time vs Audio Duration
        axes[1, 1].scatter(self.test_results['audio_durations'], self.test_results['processing_times'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Processing Time vs Audio Duration')
        axes[1, 1].set_xlabel('Audio Duration (seconds)')
        axes[1, 1].set_ylabel('Processing Time (seconds)')
        
        # Add diagonal line for real-time processing
        max_duration = max(self.test_results['audio_durations'])
        axes[1, 1].plot([0, max_duration], [0, max_duration], 'r--', alpha=0.5, label='Real-time')
        axes[1, 1].legend()
        
        # 6. Error Rate Categories
        perfect = sum(1 for wer in self.test_results['wer_scores'] if wer == 0)
        good = sum(1 for wer in self.test_results['wer_scores'] if 0 < wer <= 10)
        acceptable = sum(1 for wer in self.test_results['wer_scores'] if 10 < wer <= 25)
        poor = sum(1 for wer in self.test_results['wer_scores'] if wer > 25)
        
        categories = ['Perfect\n(0%)', 'Good\n(≤10%)', 'Acceptable\n(≤25%)', 'Poor\n(>25%)']
        counts = [perfect, good, acceptable, poor]
        colors = ['green', 'lightgreen', 'yellow', 'red']
        
        axes[1, 2].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Error Rate Categories')
        axes[1, 2].set_ylabel('Number of Samples')
        
        # Add percentage labels on bars
        total_samples = len(self.test_results['wer_scores'])
        for i, count in enumerate(counts):
            percentage = (count / total_samples) * 100
            axes[1, 2].text(i, count + 0.5, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{checkpoint_name}_detailed_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Plot: {plot_path}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _reset_results(self):
        """Reset test results storage"""
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

if __name__ == "__main__":
    print("="*80)
    print("TESTING BASE MODEL CHECKPOINTS")
    print("="*80)
    
    # Test the latest checkpoint (900 steps)
    checkpoint_path = r"C:\Poroject\base_model_checkpoints\base-checkpoint-900.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = r"C:\Poroject\base_model_checkpoints"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        exit(1)
    
    print(f"Testing base model checkpoint-900 with comprehensive analysis...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialize tester with base model
        tester = BaseModelTester(
            model_path=checkpoint_path,
            model_name="base",  # Use base model instead of tiny
            device="auto",
            language="af"
        )
        
        # Test manifest path
        test_manifest = r"C:\Poroject\processed_data\test_manifest.jsonl"
        
        if not os.path.exists(test_manifest):
            print(f"❌ Error: Test manifest file not found at {test_manifest}")
            exit(1)
        
        # Run comprehensive testing
        results = tester.test_manifest(
            manifest_path=test_manifest,
            output_dir=r"C:\Poroject\NEW SCRIPT\test_results"
        )
        
        # Save main results file
        output_file = f"base_checkpoint_900_test_results_{timestamp}.json"
        
        # Ensure results directory exists
        results_dir = r"C:\Poroject\NEW SCRIPT\test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(results_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Main results saved to: {output_path}")
        print("\n✅ Base model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing base model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)