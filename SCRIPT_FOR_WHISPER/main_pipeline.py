#!/usr/bin/env python3
"""
Whisper Training Pipeline - Main Orchestrator
Complete pipeline for data processing, feature extraction, training, validation, and testing
"""

import os
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Import our custom modules
from data_processor import DataProcessor
from feature_extractor import WhisperFeatureExtractor
from train_whisper import WhisperTrainer
from validate_whisper import WhisperValidator
from test_whisper import WhisperTester

class WhisperPipeline:
    """Complete Whisper training and evaluation pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "data": {
                "csv_path": "nchlt_xitsonga_selective_fixed.csv",
                "audio_base_path": "Data/nchlt_tso",
                "language": "en",
                "train_ratio": 0.8,
                "val_ratio": 0.1
            },
            "model": {
                "model_name": "tiny",
                "device": "auto"
            },
            "training": {
                "learning_rate": 1e-5,
                "warmup_steps": 100,
                "max_steps": 2000,
                "save_steps": 500,
                "eval_steps": 250,
                "logging_steps": 50,
                "batch_size": 4
            },
            "output": {
                "processed_data_dir": "processed_data",
                "features_dir": "features",
                "checkpoints_dir": "whisper_checkpoints",
                "validation_dir": "validation_results",
                "test_dir": "test_results"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configs (user config overrides defaults)
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
        
        return default_config
    
    def save_config(self, output_dir: str):
        """Save current configuration"""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f'pipeline_config_{self.timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
    
    def step_1_process_data(self):
        """Step 1: Data processing and splitting"""
        print("\n" + "="*60)
        print("STEP 1: DATA PROCESSING")
        print("="*60)
        
        processor = DataProcessor(
            csv_path=self.config['data']['csv_path'],
            audio_base_path=self.config['data']['audio_base_path']
        )
        
        train_data, val_data, test_data = processor.process_all(
            output_dir=self.config['output']['processed_data_dir']
        )
        
        print(f"Data processing completed!")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def step_2_extract_features(self):
        """Step 2: Feature extraction"""
        print("\n" + "="*60)
        print("STEP 2: FEATURE EXTRACTION")
        print("="*60)
        
        extractor = WhisperFeatureExtractor(
            model_name=self.config['model']['model_name'],
            device=self.config['model']['device']
        )
        
        features_dir = self.config['output']['features_dir']
        processed_data_dir = self.config['output']['processed_data_dir']
        
        # Extract features for each split
        for split in ['train', 'val', 'test']:
            manifest_path = os.path.join(processed_data_dir, f'{split}_manifest.jsonl')
            if os.path.exists(manifest_path):
                print(f"\nExtracting features for {split} set...")
                features = extractor.process_manifest(
                    manifest_path, 
                    features_dir,
                    language=self.config['data']['language']
                )
                print(f"Processed {len(features)} samples for {split} split")
        
        print("Feature extraction completed!")
    
    def step_3_train_model(self):
        """Step 3: Model training"""
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        trainer = WhisperTrainer(
            model_name=self.config['model']['model_name'],
            device=self.config['model']['device'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_steps=self.config['training']['warmup_steps'],
            max_steps=self.config['training']['max_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            logging_steps=self.config['training']['logging_steps'],
            output_dir=self.config['output']['checkpoints_dir']
        )
        
        processed_data_dir = self.config['output']['processed_data_dir']
        train_manifest = os.path.join(processed_data_dir, 'train_manifest.jsonl')
        val_manifest = os.path.join(processed_data_dir, 'val_manifest.jsonl')
        
        trainer.train(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            batch_size=self.config['training']['batch_size'],
            language=self.config['data']['language']
        )
        
        print("Model training completed!")
        return trainer
    
    def step_4_validate_model(self, model_path: str = None):
        """Step 4: Model validation"""
        print("\n" + "="*60)
        print("STEP 4: MODEL VALIDATION")
        print("="*60)
        
        if model_path is None:
            model_path = os.path.join(self.config['output']['checkpoints_dir'], 'best_model.pt')
        
        validator = WhisperValidator(
            model_path=model_path,
            model_name=self.config['model']['model_name'],
            device=self.config['model']['device'],
            language=self.config['data']['language']
        )
        
        processed_data_dir = self.config['output']['processed_data_dir']
        val_manifest = os.path.join(processed_data_dir, 'val_manifest.jsonl')
        
        validation_results = validator.validate_manifest(
            manifest_path=val_manifest,
            output_dir=self.config['output']['validation_dir']
        )
        
        print("Model validation completed!")
        return validation_results
    
    def step_5_test_model(self, model_path: str = None):
        """Step 5: Final model testing"""
        print("\n" + "="*60)
        print("STEP 5: MODEL TESTING")
        print("="*60)
        
        if model_path is None:
            model_path = os.path.join(self.config['output']['checkpoints_dir'], 'best_model.pt')
        
        tester = WhisperTester(
            model_path=model_path,
            model_name=self.config['model']['model_name'],
            device=self.config['model']['device'],
            language=self.config['data']['language']
        )
        
        processed_data_dir = self.config['output']['processed_data_dir']
        test_manifest = os.path.join(processed_data_dir, 'test_manifest.jsonl')
        
        test_results = tester.test_manifest(
            manifest_path=test_manifest,
            output_dir=self.config['output']['test_dir']
        )
        
        print("Model testing completed!")
        return test_results
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*80)
        print("WHISPER TRAINING PIPELINE - FULL EXECUTION")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Model: {self.config['model']['model_name']}")
        print(f"Language: {self.config['data']['language']}")
        print(f"Max steps: {self.config['training']['max_steps']}")
        
        # Save configuration
        self.save_config('pipeline_logs')
        
        try:
            # Step 1: Data Processing
            train_data, val_data, test_data = self.step_1_process_data()
            
            # Step 2: Feature Extraction
            self.step_2_extract_features()
            
            # Step 3: Model Training
            trainer = self.step_3_train_model()
            
            # Step 4: Model Validation
            validation_results = self.step_4_validate_model()
            
            # Step 5: Model Testing
            test_results = self.step_5_test_model()
            
            # Final Summary
            self._print_final_summary(validation_results, test_results)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            return {
                'validation_results': validation_results,
                'test_results': test_results,
                'config': self.config
            }
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_final_summary(self, validation_results, test_results):
        """Print final pipeline summary"""
        print("\n" + "="*80)
        print("FINAL PIPELINE SUMMARY")
        print("="*80)
        print(f"Model: {self.config['model']['model_name']}")
        print(f"Training steps: {self.config['training']['max_steps']}")
        print(f"Language: {self.config['data']['language']}")
        print("\nVALIDATION RESULTS:")
        print(f"  Average WER: {validation_results['average_wer']:.2f}%")
        print(f"  Samples: {validation_results['valid_samples']}")
        print("\nTEST RESULTS:")
        print(f"  Average WER: {test_results['average_wer']:.2f}%")
        print(f"  Average CER: {test_results['average_cer']:.2f}%")
        print(f"  Real-time Factor: {test_results['real_time_factor']:.2f}x")
        print(f"  Samples: {test_results['valid_samples']}")
        
        # Performance assessment
        if test_results['average_wer'] < 10:
            performance = "Excellent"
        elif test_results['average_wer'] < 20:
            performance = "Good"
        elif test_results['average_wer'] < 30:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        print(f"\nOVERALL PERFORMANCE: {performance}")

def main():
    parser = argparse.ArgumentParser(description='Whisper Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['data', 'features', 'train', 'validate', 'test', 'full'], 
                       default='full', help='Pipeline step to run')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint for validation/testing')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WhisperPipeline(args.config)
    
    # Run specified step
    if args.step == 'data':
        pipeline.step_1_process_data()
    elif args.step == 'features':
        pipeline.step_2_extract_features()
    elif args.step == 'train':
        pipeline.step_3_train_model()
    elif args.step == 'validate':
        pipeline.step_4_validate_model(args.model_path)
    elif args.step == 'test':
        pipeline.step_5_test_model(args.model_path)
    elif args.step == 'full':
        pipeline.run_full_pipeline()
    
if __name__ == "__main__":
    main()