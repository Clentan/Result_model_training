#!/usr/bin/env python3
"""
Simple End-to-End Whisper Model
Demonstrates using Whisper without external feature extraction.
This script uses only Whisper's built-in capabilities for audio processing and transcription.
"""

import whisper
import torch
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

class SimpleWhisperModel:
    """
    A simplified Whisper model that uses only Whisper's built-in feature extraction.
    No external feature extractors or custom preprocessing required.
    """
    
    def __init__(self, model_name: str = "tiny", device: Optional[str] = None):
        """
        Initialize the simple Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        
        # Load Whisper model - this includes built-in feature extraction
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"Model loaded successfully!")
        
        # Model info
        self.model_name = model_name
        print(f"Model dimensions: {self.model.dims}")
        print(f"Model vocabulary size: {self.model.dims.n_vocab} tokens")
    
    def transcribe_audio(self, audio_path: str, language: str = "af", **kwargs) -> Dict:
        """
        Transcribe audio using Whisper's built-in pipeline.
        No external feature extraction needed!
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'af' for Afrikaans)
            **kwargs: Additional Whisper transcription options
        
        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(audio_path):
            return {
                "text": "",
                "error": f"Audio file not found: {audio_path}",
                "success": False
            }
        
        try:
            start_time = time.time()
            
            # Whisper handles everything internally:
            # 1. Audio loading and preprocessing
            # 2. Feature extraction (mel spectrograms)
            # 3. Encoding and decoding
            # 4. Text generation
            result = self.model.transcribe(
                audio_path,
                language=language,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "processing_time": processing_time,
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
    
    def transcribe_batch(self, audio_files: List[str], language: str = "af", **kwargs) -> List[Dict]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Language code
            **kwargs: Additional Whisper options
        
        Returns:
            List of transcription results
        """
        results = []
        
        print(f"Transcribing {len(audio_files)} audio files...")
        
        for i, audio_path in enumerate(audio_files, 1):
            print(f"Processing {i}/{len(audio_files)}: {os.path.basename(audio_path)}")
            
            result = self.transcribe_audio(audio_path, language=language, **kwargs)
            results.append(result)
            
            if result["success"]:
                print(f"  ✓ Success: '{result['text'][:50]}...'")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def transcribe_from_manifest(self, manifest_path: str, language: str = "af", max_files: Optional[int] = None) -> List[Dict]:
        """
        Transcribe audio files from a manifest file.
        
        Args:
            manifest_path: Path to JSONL manifest file
            language: Language code
            max_files: Maximum number of files to process
        
        Returns:
            List of transcription results with ground truth comparison
        """
        if not os.path.exists(manifest_path):
            print(f"Manifest file not found: {manifest_path}")
            return []
        
        # Read manifest
        audio_files = []
        ground_truths = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_files and len(audio_files) >= max_files:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    audio_files.append(data['audio_filepath'])
                    ground_truths.append(data['text'])
                except Exception as e:
                    print(f"Error reading line {line_num}: {e}")
        
        print(f"Found {len(audio_files)} audio files in manifest")
        
        # Transcribe
        results = self.transcribe_batch(audio_files, language=language)
        
        # Add ground truth for comparison
        for i, result in enumerate(results):
            if i < len(ground_truths):
                result['ground_truth'] = ground_truths[i]
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save transcription results to JSON file.
        
        Args:
            results: List of transcription results
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """
    Demonstrate the simple Whisper model usage.
    """
    print("=" * 60)
    print("Simple End-to-End Whisper Model Demo")
    print("No external feature extraction required!")
    print("=" * 60)
    
    # Initialize model
    model = SimpleWhisperModel(model_name="tiny")
    
    print("\n" + "="*40)
    print("Key Features:")
    print("✓ Uses only Whisper's built-in capabilities")
    print("✓ No external feature extractors needed")
    print("✓ No custom preprocessing required")
    print("✓ Handles audio loading automatically")
    print("✓ Built-in mel spectrogram extraction")
    print("✓ Integrated tokenization and decoding")
    print("="*40)
    
    # Test with manifest if available
    manifest_path = "processed_data/test_manifest.jsonl"
    if os.path.exists(manifest_path):
        print(f"\nTesting with manifest: {manifest_path}")
        
        # Process a few files as demo
        results = model.transcribe_from_manifest(
            manifest_path, 
            language="af", 
            max_files=3
        )
        
        print(f"\nProcessed {len(results)} files:")
        for i, result in enumerate(results, 1):
            print(f"\nFile {i}: {os.path.basename(result.get('audio_path', 'unknown'))}")
            if result['success']:
                print(f"  Transcription: {result['text']}")
                if 'ground_truth' in result:
                    print(f"  Ground Truth:  {result['ground_truth']}")
                print(f"  Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Save results
        output_path = "simple_whisper_demo_results.json"
        model.save_results(results, output_path)
    
    else:
        print(f"\nManifest not found: {manifest_path}")
        print("You can test with individual audio files using:")
        print("  result = model.transcribe_audio('path/to/audio.wav', language='af')")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("\nThis demonstrates that Whisper can work completely")
    print("self-contained without any external feature extraction.")
    print("All audio processing, feature extraction, and transcription")
    print("is handled internally by the Whisper model.")
    print("="*60)


if __name__ == "__main__":
    main()