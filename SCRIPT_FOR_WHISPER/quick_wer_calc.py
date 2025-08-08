# Quick WER calculation from the 1000 steps model results
# Based on the first 50 samples from test_summary_20250730_170637.csv

wer_scores = [
    0.0, 20.0, 20.0, 40.0, 0.0, 80.0, 20.0, 166.67, 40.0, 140.0,
    0.0, 0.0, 40.0, 0.0, 0.0, 80.0, 80.0, 20.0, 160.0, 0.0,
    80.0, 0.0, 0.0, 120.0, 80.0, 160.0, 60.0, 0.0, 0.0, 40.0,
    80.0, 80.0, 60.0, 40.0, 20.0, 20.0, 20.0, 0.0, 60.0, 0.0,
    40.0, 0.0, 100.0, 20.0, 0.0, 40.0, 20.0, 20.0, 0.0
]

average_wer = sum(wer_scores) / len(wer_scores)
samples_below_target = len([score for score in wer_scores if score < 38.6])
percentage_below_target = (samples_below_target / len(wer_scores)) * 100

print(f"Sample size: {len(wer_scores)} (first 50 samples)")
print(f"Average WER: {average_wer:.2f}%")
print(f"Samples below 38.6%: {samples_below_target}/{len(wer_scores)} ({percentage_below_target:.1f}%)")
print(f"Target achieved: {'YES' if average_wer < 38.6 else 'NO'}")

# Count perfect scores (0% WER)
perfect_scores = len([score for score in wer_scores if score == 0.0])
print(f"Perfect transcriptions (0% WER): {perfect_scores}/{len(wer_scores)} ({perfect_scores/len(wer_scores)*100:.1f}%)")