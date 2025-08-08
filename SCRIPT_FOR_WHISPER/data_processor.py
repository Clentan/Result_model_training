import os
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data processing pipeline for Whisper training"""
    
    def __init__(self, csv_path: str, audio_base_path: str, target_sr: int = 16000):
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.target_sr = target_sr
        self.data = None
        
    def load_and_validate_data(self):
        """Load CSV data and validate audio files exist"""
        # Load and validate data
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} entries from CSV")
        
        # Validate audio files exist
        valid_data = []
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            
            # Handle path correction for nchlt dataset
            if 'nchlt_tso/audio/' in audio_path:
                # Extract just the filename
                filename = os.path.basename(audio_path)
                # Extract speaker ID from filename (e.g., nchlt_tso_016m_0091.wav -> 016)
                speaker_id = filename.split('_')[2][:3]  # Get first 3 chars after second underscore
                # Create correct path with speaker subdirectory
                audio_path = os.path.join(self.audio_base_path, 'audio', speaker_id, filename)
            elif not os.path.isabs(audio_path):
                audio_path = os.path.join(self.audio_base_path, audio_path)
            
            if os.path.exists(audio_path):
                valid_data.append({
                    'audio_path': audio_path,
                    'transcription': str(row['transcription']).strip()
                })
            else:
                print(f"File not found: {audio_path}")
        
        print(f"Valid audio files: {len(valid_data)}")
        return valid_data
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Basic text cleaning
        text = str(text).strip()
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # Shuffle data
        data_shuffled = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_total = len(data_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data_shuffled[:n_train]
        val_data = data_shuffled[n_train:n_train + n_val]
        test_data = data_shuffled[n_train + n_val:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str):
        """Save data splits to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        print(f"Data splits saved to {output_dir}")
    
    def create_manifest(self, data: pd.DataFrame, output_path: str):
        """Create manifest file for training"""
        manifest = []
        
        for _, row in data.iterrows():
            audio_path = row['audio_path']  # Already contains full corrected path
            text = self.preprocess_text(row['transcription'])
            
            # Convert to relative path from audio_base_path
            relative_path = os.path.relpath(audio_path, start=os.getcwd())
            
            try:
                duration = librosa.get_duration(filename=audio_path)
                manifest.append({
                    'audio_filepath': relative_path,
                    'text': text,
                    'duration': duration
                })
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in manifest:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Manifest saved to {output_path}")
    
    def process_all(self, output_dir: str = 'processed_data'):
        """Complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load and validate data
        valid_data = self.load_and_validate_data()
        self.data = pd.DataFrame(valid_data)
        
        # Clean text
        self.data['transcription'] = self.data['transcription'].apply(self.preprocess_text)
        
        # Split data
        train_data, val_data, test_data = self.split_data()
        
        # Save splits
        self.save_splits(train_data, val_data, test_data, output_dir)
        
        # Create manifests
        self.create_manifest(train_data, os.path.join(output_dir, 'train_manifest.jsonl'))
        self.create_manifest(val_data, os.path.join(output_dir, 'val_manifest.jsonl'))
        self.create_manifest(test_data, os.path.join(output_dir, 'test_manifest.jsonl'))
        
        print("Data processing completed!")
        
        return train_data, val_data, test_data

if __name__ == "__main__":
    # Configuration
    csv_file = "nchlt_xitsonga_selective_fixed.csv"
    audio_dir = "Data/nchlt.speech.corpus.tso/nchlt_tso"  # Base directory containing audio files
    output_dir = "processed_data"
    
    # Example usage
    processor = DataProcessor(
        csv_path=csv_file,
        audio_base_path=audio_dir
    )
    
    # Load and validate data
    valid_data = processor.load_and_validate_data()
    
    if len(valid_data) == 0:
        print("No valid audio files found. Please check your data paths.")
        exit()
    
    # Convert to DataFrame for processing
    processor.data = pd.DataFrame(valid_data)
    
    # Preprocess text
    processor.data['transcription'] = processor.data['transcription'].apply(
        processor.preprocess_text
    )
    
    # Split data
    train_data, val_data, test_data = processor.split_data()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    processor.save_splits(train_data, val_data, test_data, output_dir)
    
    # Create manifests
    processor.create_manifest(train_data, os.path.join(output_dir, 'train_manifest.jsonl'))
    processor.create_manifest(val_data, os.path.join(output_dir, 'val_manifest.jsonl'))
    processor.create_manifest(test_data, os.path.join(output_dir, 'test_manifest.jsonl'))
    
    print("Data processing completed successfully!")