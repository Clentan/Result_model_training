import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import json

def load_validation_data(file_path):
    """Load validation data from CSV file and calculate summary statistics"""
    df = pd.read_csv(file_path)
    
    # Calculate summary statistics
    summary = {
        'total_samples': len(df),
        'average_wer': df['wer_score'].mean(),
        'std_wer': df['wer_score'].std(),
        'median_wer': df['wer_score'].median(),
        'min_wer': df['wer_score'].min(),
        'max_wer': df['wer_score'].max(),
        'average_processing_time': df['processing_time'].mean(),
        'std_processing_time': df['processing_time'].std(),
        'median_processing_time': df['processing_time'].median()
    }
    
    return df, summary

def create_wer_distribution_plot(df, ax):
    """Create WER distribution histogram"""
    ax.hist(df['wer_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('WER Score (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('WER Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_wer = df['wer_score'].mean()
    std_wer = df['wer_score'].std()
    ax.axvline(mean_wer, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_wer:.1f}%')
    ax.legend()

def create_processing_time_plot(df, ax):
    """Create processing time distribution"""
    ax.hist(df['processing_time'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Processing Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Processing Time Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_time = df['processing_time'].mean()
    ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f}s')
    ax.legend()

def create_wer_vs_processing_time_plot(df, ax):
    """Create scatter plot of WER vs Processing Time"""
    ax.scatter(df['processing_time'], df['wer_score'], alpha=0.6, color='purple')
    ax.set_xlabel('Processing Time (seconds)')
    ax.set_ylabel('WER Score (%)')
    ax.set_title('WER vs Processing Time')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['processing_time'].corr(df['wer_score'])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_summary_stats_plot(summary, ax):
    """Create summary statistics bar plot"""
    stats = {
        'Avg WER (%)': summary['average_wer'],
        'Std WER (%)': summary['std_wer'],
        'Median WER (%)': summary['median_wer'],
        'Avg Proc Time (s)': summary['average_processing_time'],
        'Std Proc Time (s)': summary['std_processing_time']
    }
    
    bars = ax.bar(range(len(stats)), list(stats.values()), 
                  color=['lightcoral', 'lightblue', 'lightgreen', 'gold', 'plum'])
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(list(stats.keys()), rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Summary Statistics')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stats.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

def create_comprehensive_validation_analysis(csv_file_path, output_image_path):
    """Create comprehensive validation analysis visualization"""
    # Load data
    df, summary = load_validation_data(csv_file_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comprehensive Validation Analysis\nTotal Samples: {summary["total_samples"]} | Average WER: {summary["average_wer"]:.1f}%', 
                 fontsize=16, fontweight='bold')
    
    # Create plots
    create_wer_distribution_plot(df, axes[0, 0])
    create_processing_time_plot(df, axes[0, 1])
    create_wer_vs_processing_time_plot(df, axes[1, 0])
    create_summary_stats_plot(summary, axes[1, 1])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive validation analysis saved to: {output_image_path}")
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Average WER: {summary['average_wer']:.2f}%")
    print(f"WER Standard Deviation: {summary['std_wer']:.2f}%")
    print(f"Median WER: {summary['median_wer']:.2f}%")
    print(f"WER Range: {summary['min_wer']:.1f}% - {summary['max_wer']:.1f}%")
    print(f"Average Processing Time: {summary['average_processing_time']:.2f}s")
    print(f"Processing Time Std Dev: {summary['std_processing_time']:.2f}s")
    
    return summary

if __name__ == "__main__":
    # File paths
    csv_file = "validation_results/validation_summary_20250801_161310.csv"
    output_image = "comprehensive_validation_analysis_20250801.png"
    
    # Create comprehensive analysis
    summary = create_comprehensive_validation_analysis(csv_file, output_image)
    
    plt.show()