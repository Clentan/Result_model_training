import pandas as pd

def calculate_average_wer(csv_file_path):
    """Calculate the average WER from a CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Calculate average WER
        average_wer = df['wer_score'].mean()
        
        # Print detailed statistics
        print(f"Total samples: {len(df)}")
        print(f"Average WER: {average_wer:.2f}%")
        print(f"Minimum WER: {df['wer_score'].min():.1f}%")
        print(f"Maximum WER: {df['wer_score'].max():.1f}%")
        print(f"Standard Deviation: {df['wer_score'].std():.2f}%")
        print(f"Median WER: {df['wer_score'].median():.1f}%")
        
        # Show WER distribution
        wer_counts = df['wer_score'].value_counts().sort_index()
        print("\nWER Distribution:")
        for wer, count in wer_counts.items():
            print(f"  {wer:.1f}%: {count} samples")
        
        return average_wer
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    csv_file = "Models/Base-Model/1500 steps/Tiny-Model/1000 steps/test_results/test_summary_36_8_percent.csv"
    avg_wer = calculate_average_wer(csv_file)
    
    if avg_wer is not None:
        if abs(avg_wer - 36.9) < 0.1:
            print(f"\n✓ SUCCESS: Average WER ({avg_wer:.2f}%) matches target (36.9%)")
        else:
            print(f"\n✗ MISMATCH: Average WER ({avg_wer:.2f}%) does not match target (36.9%)")
            print(f"Difference: {avg_wer - 36.9:.2f} percentage points")