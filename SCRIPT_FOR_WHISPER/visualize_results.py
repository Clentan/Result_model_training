import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_test_results(file_path):
    """Load test results from JSON or CSV file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Convert CSV to JSON-like structure
        detailed_results = []
        for _, row in df.iterrows():
            detailed_results.append({
                'audio_path': row['audio_path'],
                'reference': row['reference'],
                'prediction': row['prediction'],
                'wer_score': row['wer_score'],
                'cer_score': row['cer_score'],
                'processing_time': row['processing_time'],
                'audio_duration': row['audio_duration'],
                'word_count': row['word_count'],
                'char_count': row['char_count']
            })
        
        # Calculate summary statistics
        summary = {
            'total_samples': len(df),
            'average_wer': df['wer_score'].mean(),
            'average_cer': df['cer_score'].mean(),
            'min_wer': df['wer_score'].min(),
            'max_wer': df['wer_score'].max(),
            'min_cer': df['cer_score'].min(),
            'max_cer': df['cer_score'].max(),
            'std_wer': df['wer_score'].std(),
            'std_cer': df['cer_score'].std(),
            'median_wer': df['wer_score'].median(),
            'median_cer': df['cer_score'].median(),
            'total_processing_time': df['processing_time'].sum(),
            'total_audio_duration': df['audio_duration'].sum(),
            'real_time_factor': df['processing_time'].sum() / df['audio_duration'].sum(),
            'model_type': 'Fine-tuned',
            'model_name': 'tiny'
        }
        
        return {
            'summary': summary,
            'detailed_results': detailed_results
        }

def create_wer_distribution_plot(data):
    """Create WER distribution histogram"""
    wer_scores = [result['wer_score'] for result in data['detailed_results']]
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.subplot(2, 2, 1)
    plt.hist(wer_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('WER Score Distribution')
    plt.xlabel('WER Score (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    avg_wer = data['summary']['average_wer']
    plt.axvline(avg_wer, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_wer:.1f}%')
    plt.legend()
    
    return plt

def create_cer_distribution_plot(data, plt_obj):
    """Create CER distribution histogram"""
    cer_scores = [result['cer_score'] for result in data['detailed_results']]
    
    plt_obj.subplot(2, 2, 2)
    plt_obj.hist(cer_scores, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt_obj.title('CER Score Distribution')
    plt_obj.xlabel('CER Score (%)')
    plt_obj.ylabel('Frequency')
    plt_obj.grid(True, alpha=0.3)
    
    # Add statistics text
    avg_cer = data['summary']['average_cer']
    plt_obj.axvline(avg_cer, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_cer:.1f}%')
    plt_obj.legend()
    
    return plt_obj

def create_processing_time_plot(data, plt_obj):
    """Create processing time vs audio duration scatter plot"""
    processing_times = [result['processing_time'] for result in data['detailed_results']]
    audio_durations = [result['audio_duration'] for result in data['detailed_results']]
    
    plt_obj.subplot(2, 2, 3)
    plt_obj.scatter(audio_durations, processing_times, alpha=0.6, color='green')
    plt_obj.title('Processing Time vs Audio Duration')
    plt_obj.xlabel('Audio Duration (seconds)')
    plt_obj.ylabel('Processing Time (seconds)')
    plt_obj.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(audio_durations, processing_times, 1)
    p = np.poly1d(z)
    plt_obj.plot(audio_durations, p(audio_durations), "r--", alpha=0.8)
    
    return plt_obj

def create_summary_stats_plot(data, plt_obj):
    """Create summary statistics bar chart"""
    summary = data['summary']
    
    plt_obj.subplot(2, 2, 4)
    
    # Create summary statistics
    stats = {
        'Avg WER': summary['average_wer'],
        'Avg CER': summary['average_cer'],
        'Min WER': summary['min_wer'],
        'Max WER': min(summary['max_wer'], 200),  # Cap at 200 for better visualization
        'Std WER': summary['std_wer']
    }
    
    bars = plt_obj.bar(stats.keys(), stats.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
    plt_obj.title('Summary Statistics')
    plt_obj.ylabel('Score (%)')
    plt_obj.xticks(rotation=45)
    plt_obj.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stats.values()):
        plt_obj.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom')
    
    return plt_obj

def create_comprehensive_visualization(json_path, output_path=None):
    """Create comprehensive visualization of test results"""
    # Load data
    data = load_test_results(json_path)
    
    # Create the main plot
    plt_obj = create_wer_distribution_plot(data)
    plt_obj = create_cer_distribution_plot(data, plt_obj)
    plt_obj = create_processing_time_plot(data, plt_obj)
    plt_obj = create_summary_stats_plot(data, plt_obj)
    
    # Add main title
    plt_obj.suptitle(f'Whisper Model Test Results Analysis\nAverage WER: {data["summary"]["average_wer"]:.1f}% | Total Samples: {data["summary"]["total_samples"]}', 
                    fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt_obj.tight_layout()
    plt_obj.subplots_adjust(top=0.9)
    
    # Save plot
    if output_path is None:
        output_path = 'test_results_visualization.png'
    
    plt_obj.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt_obj.show()
    
    return plt_obj

if __name__ == "__main__":
    # Use the new CSV file with 36.9% WER
    csv_file = "C:/Poroject/Models/Base-Model/1500 steps/Tiny-Model/1000 steps/test_results/test_summary_36_8_percent.csv"
    
    # Create comprehensive visualization
    create_comprehensive_visualization(csv_file, "comprehensive_test_analysis_36_9_percent.png")

    print("\n=== Test Results Summary ===")
    data = load_test_results(csv_file)
    summary = data['summary']
    
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Average WER: {summary['average_wer']:.2f}%")
    print(f"Average CER: {summary['average_cer']:.2f}%")
    print(f"WER Range: {summary['min_wer']:.1f}% - {summary['max_wer']:.1f}%")
    print(f"Processing Speed: {summary['real_time_factor']:.3f}x real-time")
    print(f"Model Type: {summary['model_type']} {summary['model_name']}")