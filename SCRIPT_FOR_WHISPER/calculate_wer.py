import pandas as pd
import sys

def calculate_average_wer(csv_file_path):
    """Calculate average WER from test results CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Calculate average WER
        average_wer = df['wer_score'].mean()
        median_wer = df['wer_score'].median()
        min_wer = df['wer_score'].min()
        max_wer = df['wer_score'].max()
        
        # Count samples with WER < 38.6%
        samples_below_target = (df['wer_score'] < 38.6).sum()
        total_samples = len(df)
        percentage_below_target = (samples_below_target / total_samples) * 100
        
        print(f"\n=== WER Analysis for {csv_file_path} ===")
        print(f"Total samples: {total_samples}")
        print(f"Average WER: {average_wer:.2f}%")
        print(f"Median WER: {median_wer:.2f}%")
        print(f"Min WER: {min_wer:.2f}%")
        print(f"Max WER: {max_wer:.2f}%")
        print(f"\nTarget Analysis (WER < 38.6%):")
        print(f"Samples meeting target: {samples_below_target}/{total_samples} ({percentage_below_target:.1f}%)")
        print(f"Target achieved: {'YES' if average_wer < 38.6 else 'NO'}")
        
        return average_wer
        
    except Exception as e:
        print(f"Error reading file {csv_file_path}: {e}")
        return None

if __name__ == "__main__":
    # Test different model results
    models_to_test = [
        "Tiny-Model/1000 steps/test_results/test_summary_20250730_170637.csv",
        "Tiny-Model/1500 steps/test_results/test_summary_20250730_201751.csv",
        "Tiny-Model/2000 steps/test_results/test_summary_20250730_193855.csv"
    ]
    
    best_wer = float('inf')
    best_model = None
    
    for model_path in models_to_test:
        try:
            wer = calculate_average_wer(model_path)
            if wer is not None and wer < best_wer:
                best_wer = wer
                best_model = model_path
        except FileNotFoundError:
            print(f"File not found: {model_path}")
            continue
    
    print(f"\n=== SUMMARY ===")
    print(f"Best performing model: {best_model}")
    print(f"Best average WER: {best_wer:.2f}%")
    print(f"Target WER < 38.6%: {'ACHIEVED' if best_wer < 38.6 else 'NOT ACHIEVED'}")