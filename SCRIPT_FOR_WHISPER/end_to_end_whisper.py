import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import whisper
import numpy as np
import json
import librosa
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import jiwer
import warnings
warnings.filterwarnings('ignore')

class EndToEndWhisperModel(nn.Module):
    """
    End-to-End Whisper Model that handles everything from raw audio input to text output.
    This model integrates audio preprocessing, feature extraction, and transcription in a single pipeline.
    """
    
    def __init__(self, 
                 model_name: str = "tiny",
                 device: str = "auto",
                 language: str = "en",
                 checkpoint_path: str = None):
        super().__init__()
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.language = language
        
        # Load Whisper model
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading fine-tuned model from {checkpoint_path}")
            self.whisper_model = whisper.load_model(model_name, device=self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.whisper_model.load_state_dict(checkpoint['model_state_dict'])
            self.is_finetuned = True
        else:
            print(f"Loading pre-trained Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name, device=self.device)
            self.is_finetuned = False
        
        # Audio preprocessing parameters
        self.sample_rate = 16000
        self.max_duration = 30  # seconds
        self.n_mels = 80
        self.hop_length = 160
        self.n_fft = 400
        
        print(f"End-to-End Whisper Model initialized on {self.device}")
        print(f"Model: {'Fine-tuned' if self.is_finetuned else 'Pre-trained'} {model_name}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_and_preprocess_audio(self, audio_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load and preprocess audio from file path or numpy array.
        
        Args:
            audio_input: Either file path (str) or audio array (np.ndarray)
            
        Returns:
            Preprocessed audio array
        """
        try:
            if isinstance(audio_input, str):
                # Load from file
                audio, _ = librosa.load(audio_input, sr=self.sample_rate, mono=True)
            else:
                # Use provided array
                audio = audio_input.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Pad or trim to max duration
            max_length = self.sample_rate * self.max_duration
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            
            return audio
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return np.zeros(self.sample_rate * self.max_duration, dtype=np.float32)
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract mel spectrogram features using Whisper's preprocessing.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Mel spectrogram tensor
        """
        try:
            # Use Whisper's built-in preprocessing
            audio_tensor = torch.from_numpy(audio).float()
            mel = whisper.log_mel_spectrogram(audio_tensor).to(self.device)
            return mel
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return dummy features
            return torch.zeros((self.n_mels, 3000), device=self.device)
    
    def transcribe(self, 
                   audio_input: Union[str, np.ndarray],
                   return_timestamps: bool = False,
                   return_confidence: bool = False) -> Dict[str, Union[str, float, List]]:
        """
        End-to-end transcription from audio input to text output.
        
        Args:
            audio_input: Audio file path or numpy array
            return_timestamps: Whether to return word-level timestamps
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing transcription and optional metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess audio
            audio = self.load_and_preprocess_audio(audio_input)
            
            # Step 2: Extract features
            mel_features = self.extract_features(audio)
            
            # Step 3: Transcribe using Whisper
            self.whisper_model.eval()
            with torch.no_grad():
                # Use Whisper's decode function for end-to-end transcription
                options = whisper.DecodingOptions(
                    language=self.language,
                    task="transcribe",
                    fp16=True if self.device == "cuda" else False,
                    temperature=0.0,  # Deterministic output
                    beam_size=5,  # Beam search for better quality
                    best_of=5,  # Multiple candidates
                    without_timestamps=not return_timestamps
                )
                
                result = whisper.decode(self.whisper_model, mel_features, options)
                
            processing_time = time.time() - start_time
            
            # Prepare output
            output = {
                'text': result.text.strip(),
                'processing_time': processing_time,
                'audio_duration': len(audio) / self.sample_rate,
                'language': result.language if hasattr(result, 'language') else self.language
            }
            
            if return_timestamps and hasattr(result, 'segments'):
                output['segments'] = result.segments
            
            if return_confidence and hasattr(result, 'avg_logprob'):
                output['confidence'] = result.avg_logprob
            
            return output
            
        except Exception as e:
            print(f"Error in end-to-end transcription: {e}")
            return {
                'text': '',
                'processing_time': time.time() - start_time,
                'audio_duration': 0.0,
                'language': self.language,
                'error': str(e)
            }
    
    def batch_transcribe(self, 
                        audio_inputs: List[Union[str, np.ndarray]],
                        batch_size: int = 4) -> List[Dict]:
        """
        Batch transcription for multiple audio inputs.
        
        Args:
            audio_inputs: List of audio file paths or numpy arrays
            batch_size: Number of audio files to process simultaneously
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i in range(0, len(audio_inputs), batch_size):
            batch = audio_inputs[i:i + batch_size]
            batch_results = []
            
            for audio_input in batch:
                result = self.transcribe(audio_input)
                batch_results.append(result)
            
            results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{(len(audio_inputs) + batch_size - 1)//batch_size}")
        
        return results
    
    def evaluate_on_dataset(self, 
                           manifest_path: str,
                           output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate the end-to-end model on a dataset.
        
        Args:
            manifest_path: Path to manifest file with audio paths and reference texts
            output_dir: Directory to save evaluation results
            
        Returns:
            Evaluation metrics and results
        """
        print(f"Starting end-to-end evaluation on {manifest_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = [json.loads(line.strip()) for line in f]
        
        results = {
            'predictions': [],
            'references': [],
            'wer_scores': [],
            'cer_scores': [],
            'processing_times': [],
            'audio_durations': [],
            'audio_paths': []
        }
        
        print(f"Evaluating {len(manifest_data)} samples...")
        
        for i, item in enumerate(manifest_data):
            audio_path = item['audio_filepath']
            reference_text = item['text']
            
            print(f"Processing {i+1}/{len(manifest_data)}: {os.path.basename(audio_path)}")
            
            # End-to-end transcription
            transcription_result = self.transcribe(audio_path)
            prediction = transcription_result['text']
            
            # Calculate metrics
            try:
                wer = jiwer.wer(reference_text.lower(), prediction.lower()) * 100
                cer = jiwer.cer(reference_text.lower(), prediction.lower()) * 100
            except:
                wer, cer = 100.0, 100.0
            
            # Store results
            results['predictions'].append(prediction)
            results['references'].append(reference_text)
            results['wer_scores'].append(wer)
            results['cer_scores'].append(cer)
            results['processing_times'].append(transcription_result['processing_time'])
            results['audio_durations'].append(transcription_result['audio_duration'])
            results['audio_paths'].append(audio_path)
            
            print(f"  WER: {wer:.1f}%, CER: {cer:.1f}%, Time: {transcription_result['processing_time']:.2f}s")
        
        # Calculate summary metrics
        avg_wer = np.mean(results['wer_scores'])
        avg_cer = np.mean(results['cer_scores'])
        avg_processing_time = np.mean(results['processing_times'])
        total_audio_duration = sum(results['audio_durations'])
        rtf = sum(results['processing_times']) / total_audio_duration if total_audio_duration > 0 else 0
        
        summary = {
            'average_wer': avg_wer,
            'average_cer': avg_cer,
            'std_wer': np.std(results['wer_scores']),
            'std_cer': np.std(results['cer_scores']),
            'min_wer': min(results['wer_scores']),
            'max_wer': max(results['wer_scores']),
            'average_processing_time': avg_processing_time,
            'total_processing_time': sum(results['processing_times']),
            'total_audio_duration': total_audio_duration,
            'real_time_factor': rtf,
            'num_samples': len(manifest_data),
            'model_type': 'Fine-tuned' if self.is_finetuned else 'Pre-trained',
            'model_name': self.model_name
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"end_to_end_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, f"end_to_end_summary_{timestamp}.csv")
        import pandas as pd
        df = pd.DataFrame({
            'audio_path': results['audio_paths'],
            'reference': results['references'],
            'prediction': results['predictions'],
            'wer_score': results['wer_scores'],
            'cer_score': results['cer_scores'],
            'processing_time': results['processing_times'],
            'audio_duration': results['audio_durations']
        })
        df.to_csv(csv_file, index=False)
        
        # Generate visualization
        self._generate_evaluation_plots(results, summary, output_dir, timestamp)
        
        print(f"\nEnd-to-End Evaluation Results:")
        print(f"Average WER: {avg_wer:.2f}%")
        print(f"Average CER: {avg_cer:.2f}%")
        print(f"Real-time Factor: {rtf:.2f}x")
        print(f"Results saved to {output_dir}")
        
        return summary
    
    def _generate_evaluation_plots(self, results: Dict, summary: Dict, output_dir: str, timestamp: str):
        """Generate evaluation visualization plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # WER distribution
        ax1.hist(results['wer_scores'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(summary['average_wer'], color='red', linestyle='--', linewidth=2, label=f'Mean: {summary["average_wer"]:.1f}%')
        ax1.set_xlabel('Word Error Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('WER Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CER distribution
        ax2.hist(results['cer_scores'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(summary['average_cer'], color='red', linestyle='--', linewidth=2, label=f'Mean: {summary["average_cer"]:.1f}%')
        ax2.set_xlabel('Character Error Rate (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CER Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Processing time vs Audio duration
        ax3.scatter(results['audio_durations'], results['processing_times'], alpha=0.6, color='purple')
        ax3.plot([0, max(results['audio_durations'])], [0, max(results['audio_durations'])], 'r--', label='Real-time line')
        ax3.set_xlabel('Audio Duration (s)')
        ax3.set_ylabel('Processing Time (s)')
        ax3.set_title(f'Processing Efficiency (RTF: {summary["real_time_factor"]:.2f}x)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # WER vs Processing time
        ax4.scatter(results['processing_times'], results['wer_scores'], alpha=0.6, color='orange')
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Word Error Rate (%)')
        ax4.set_title('WER vs Processing Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"end_to_end_evaluation_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {plot_file}")
    
    def forward(self, mel_features: torch.Tensor, tokens: torch.Tensor = None):
        """
        Forward pass for training (if needed).
        
        Args:
            mel_features: Mel spectrogram features
            tokens: Target tokens for training
            
        Returns:
            Model output
        """
        if tokens is not None:
            # Training mode
            return self.whisper_model(mel_features, tokens)
        else:
            # Inference mode
            return self.whisper_model.encoder(mel_features)


class EndToEndWhisperDataset(Dataset):
    """Dataset class for end-to-end Whisper training"""
    
    def __init__(self, manifest_path: str, model: EndToEndWhisperModel, language: str = "en"):
        self.manifest_path = manifest_path
        self.model = model
        self.language = language
        self.data = self._load_manifest()
    
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
        
        # Load and preprocess audio
        audio = self.model.load_and_preprocess_audio(item['audio_filepath'])
        
        # Extract features
        mel_features = self.model.extract_features(audio)
        
        # Tokenize text
        tokens = self.model.whisper_model.tokenizer.encode(item['text'])
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'mel_features': mel_features,
            'tokens': tokens,
            'text': item['text'],
            'audio_path': item['audio_filepath']
        }


if __name__ == "__main__":
    # Example usage of End-to-End Whisper Model
    
    print("=== End-to-End Whisper Model Demo ===")
    
    # Initialize model
    model = EndToEndWhisperModel(
        model_name="tiny",
        checkpoint_path="whisper_checkpoints/checkpoint-1000.pt",  # Use fine-tuned model if available
        language="en"
    )
    
    # Example 1: Single audio transcription
    print("\n1. Single Audio Transcription:")
    # Note: Replace with actual audio file path
    audio_file = "Data/nchlt_tso/audio/nchlt_tso_001.wav"  # Example path
    if os.path.exists(audio_file):
        result = model.transcribe(audio_file, return_timestamps=True, return_confidence=True)
        print(f"Transcription: {result['text']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Audio duration: {result['audio_duration']:.2f}s")
    else:
        print("Audio file not found - skipping single transcription demo")
    
    # Example 2: Batch transcription
    print("\n2. Batch Transcription Demo:")
    audio_files = ["Data/nchlt_tso/audio/nchlt_tso_001.wav", "Data/nchlt_tso/audio/nchlt_tso_002.wav"]
    existing_files = [f for f in audio_files if os.path.exists(f)]
    if existing_files:
        batch_results = model.batch_transcribe(existing_files, batch_size=2)
        for i, result in enumerate(batch_results):
            print(f"File {i+1}: {result['text'][:50]}...")
    else:
        print("No audio files found - skipping batch demo")
    
    # Example 3: Dataset evaluation
    print("\n3. Dataset Evaluation:")
    test_manifest = "processed_data/test_manifest.jsonl"
    if os.path.exists(test_manifest):
        evaluation_results = model.evaluate_on_dataset(
            manifest_path=test_manifest,
            output_dir="end_to_end_results"
        )
        print(f"\nEvaluation completed!")
        print(f"Final WER: {evaluation_results['average_wer']:.2f}%")
        print(f"Final CER: {evaluation_results['average_cer']:.2f}%")
        print(f"Real-time Factor: {evaluation_results['real_time_factor']:.2f}x")
    else:
        print("Test manifest not found - skipping evaluation demo")
    
    print("\n=== End-to-End Whisper Model Demo Completed ===")