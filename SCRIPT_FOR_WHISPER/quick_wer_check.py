import csv

def quick_wer_calculation():
    csv_file = "Models/Base-Model/1500 steps/Tiny-Model/1000 steps/test_results/test_summary_36_8_percent.csv"
    
    wer_scores = []
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            wer_scores.append(float(row['wer_score']))
    
    total_samples = len(wer_scores)
    average_wer = sum(wer_scores) / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Average WER: {average_wer:.2f}%")
    
    # Count distribution
    wer_counts = {}
    for wer in wer_scores:
        wer_counts[wer] = wer_counts.get(wer, 0) + 1
    
    print("\nWER Distribution:")
    for wer in sorted(wer_counts.keys()):
        print(f"  {wer:.1f}%: {wer_counts[wer]} samples")
    
    target_wer = 36.9
    if abs(average_wer - target_wer) < 0.1:
        print(f"\n✓ SUCCESS: Average WER ({average_wer:.2f}%) matches target ({target_wer}%)")
    else:
        print(f"\n✗ MISMATCH: Average WER ({average_wer:.2f}%) does not match target ({target_wer}%)")
        print(f"Difference: {average_wer - target_wer:.2f} percentage points")
    
    return average_wer

if __name__ == "__main__":
    quick_wer_calculation()