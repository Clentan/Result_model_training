#!/usr/bin/env python3
"""
Quick Base Model Test
Tests a few samples from the base model checkpoint to quickly evaluate performance.
"""

import os
import torch
import whisper
import json
import jiwer
import time
from datetime import datetime

def quick_test_base_model():
    """Quick test of base model with a few samples"""
    print("="*60)
    print("QUICK BASE MODEL TEST")
    print("="*60)
    
    # Checkpoint path
    checkpoint_path = r"C:\Poroject\base_model_checkpoints\base-checkpoint-900.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📁 Loading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device: {device}")
    
    try:
        # Load base model
        model = whisper.load_model("base", device=device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully")
        
        # Load test manifest
        test_manifest = r"C:\Poroject\processed_data\test_manifest.jsonl"
        
        if not os.path.exists(test_manifest):
            print(f"❌ Test manifest not found: {test_manifest}")
            return
        
        # Load manifest data
        with open(test_manifest, 'r', encoding='utf-8') as f:
            manifest_data = [json.loads(line) for line in f]
        
        # Test only first 10 samples for quick evaluation
        test_samples = manifest_data[:10]
        print(f"\n🔍 Testing {len(test_samples)} samples...")
        
        results = []
        total_wer = 0
        total_cer = 0
        
        for i, sample in enumerate(test_samples):
            audio_path = sample['audio_filepath']
            reference = sample['text']
            
            print(f"\n📝 Sample {i+1}/{len(test_samples)}")
            print(f"Audio: {os.path.basename(audio_path)}")
            
            # Handle relative paths
            if not os.path.isabs(audio_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                audio_path = os.path.join(project_root, audio_path)
            
            try:
                # Transcribe with simpler options for speed
                start_time = time.time()
                result = whisper.transcribe(
                    model, 
                    audio_path, 
                    language="af",
                    task="transcribe",
                    verbose=False,
                    temperature=0.0,
                    beam_size=1,  # Faster decoding
                    best_of=1     # Faster decoding
                )
                processing_time = time.time() - start_time
                
                prediction = result['text'].strip()
                
                # Calculate metrics
                try:
                    wer = jiwer.wer(reference, prediction) * 100
                    cer = jiwer.cer(reference, prediction) * 100
                except:
                    wer = 100.0
                    cer = 100.0
                
                total_wer += wer
                total_cer += cer
                
                # Store result
                result_data = {
                    'audio_path': audio_path,
                    'reference': reference,
                    'prediction': prediction,
                    'wer': wer,
                    'cer': cer,
                    'processing_time': processing_time
                }
                results.append(result_data)
                
                # Print results
                print(f"Reference: {reference}")
                print(f"Prediction: {prediction}")
                print(f"WER: {wer:.1f}% | CER: {cer:.1f}% | Time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {e}")
                continue
        
        # Calculate averages
        if results:
            avg_wer = total_wer / len(results)
            avg_cer = total_cer / len(results)
            
            print("\n" + "="*60)
            print("📊 QUICK TEST RESULTS")
            print("="*60)
            print(f"🎯 Samples tested: {len(results)}")
            print(f"📈 Average WER: {avg_wer:.2f}%")
            print(f"📈 Average CER: {avg_cer:.2f}%")
            
            # Count perfect predictions
            perfect = sum(1 for r in results if r['wer'] == 0)
            good = sum(1 for r in results if r['wer'] <= 10)
            
            print(f"✨ Perfect predictions (0% WER): {perfect}/{len(results)} ({perfect/len(results)*100:.1f}%)")
            print(f"👍 Good predictions (≤10% WER): {good}/{len(results)} ({good/len(results)*100:.1f}%)")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"quick_base_test_results_{timestamp}.json"
            output_dir = r"C:\Poroject\NEW SCRIPT\test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, output_file)
            
            summary = {
                'model_info': {
                    'checkpoint': os.path.basename(checkpoint_path),
                    'model_type': 'base',
                    'device': device,
                    'language': 'af'
                },
                'test_summary': {
                    'samples_tested': len(results),
                    'avg_wer': avg_wer,
                    'avg_cer': avg_cer,
                    'perfect_predictions': perfect,
                    'good_predictions': good
                },
                'individual_results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_path}")
            print("\n✅ Quick test completed successfully!")
            
        else:
            print("\n❌ No samples were successfully processed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test_base_model()