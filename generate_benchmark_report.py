#!/usr/bin/env python3
"""
Benchmark Comparison Report Generator
Generates comprehensive visual reports comparing GPU (Metal) vs DuckDB performance
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class BenchmarkData:
    """Container for benchmark data"""
    
    def __init__(self):
        # SF-1 Data (1GB)
        self.sf1_queries = [
            'Selection\n< 1000',
            'Selection\n< 10000',
            'Selection\n< 50000',
            'SUM\n(l_quantity)',
            'Hash\nJoin',
            'TPC-H\nQuery 1',
            'TPC-H\nQuery 3',
            'TPC-H\nQuery 6',
            'TPC-H\nQuery 9',
            'TPC-H\nQuery 13'
        ]
        
        self.sf1_gpu_times = [0.884, 0.900, 0.946, 1.325, 12.335, 39.660, 12.744, 4.480, 40.480, 34.969]
        self.sf1_duckdb_times = [38.000, 37.190, 36.610, 35.340, 75.150, 84.140, 58.640, 45.850, 110.260, 103.720]
        self.sf1_speedup = [42.99, 41.33, 38.69, 26.68, 6.09, 2.12, 4.60, 10.23, 2.72, 2.97]
        
        # SF-10 Data (10GB)
        self.sf10_queries = self.sf1_queries  # Same queries
        
        self.sf10_gpu_times = [10.172, 8.003, 8.133, 5.050, 62.510, 331.301, 65.985, 124.270, 826.500, 268.669]
        self.sf10_duckdb_times = [189.090, 85.030, 84.900, 103.950, 481.640, 562.720, 303.830, 162.420, 821.980, 759.030]
        self.sf10_speedup = [18.59, 10.62, 10.44, 20.59, 7.71, 1.70, 4.60, 1.31, 0.99, 2.83]

def plot_execution_time_comparison(data, output_dir):
    """Create side-by-side comparison of execution times for SF-1 and SF-10"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(data.sf1_queries))
    width = 0.35
    
    # SF-1 Plot
    bars1 = ax1.bar(x - width/2, data.sf1_gpu_times, width, label='GPU (Metal)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, data.sf1_duckdb_times, width, label='DuckDB', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('SF-1 Dataset (1GB) - Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # SF-10 Plot
    bars3 = ax2.bar(x - width/2, data.sf10_gpu_times, width, label='GPU (Metal)', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, data.sf10_duckdb_times, width, label='DuckDB', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('SF-10 Dataset (10GB) - Execution Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data.sf10_queries, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: execution_time_comparison.png")
    plt.close()

def plot_speedup_analysis(data, output_dir):
    """Create speedup comparison charts"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(data.sf1_queries))
    
    # SF-1 Speedup
    colors1 = ['#e74c3c' if s < 1 else '#27ae60' for s in data.sf1_speedup]
    bars1 = axes[0].bar(x, data.sf1_speedup, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
    axes[0].set_xlabel('Query Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Speedup Factor (×)', fontsize=12, fontweight='bold')
    axes[0].set_title('SF-1 Dataset (1GB) - GPU Speedup over DuckDB', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, data.sf1_speedup)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # SF-10 Speedup
    colors2 = ['#e74c3c' if s < 1 else '#27ae60' for s in data.sf10_speedup]
    bars2 = axes[1].bar(x, data.sf10_speedup, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
    axes[1].set_xlabel('Query Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Speedup Factor (×)', fontsize=12, fontweight='bold')
    axes[1].set_title('SF-10 Dataset (10GB) - GPU Speedup over DuckDB', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(data.sf10_queries, rotation=45, ha='right')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, data.sf10_speedup)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: speedup_analysis.png")
    plt.close()

def plot_log_scale_comparison(data, output_dir):
    """Create log-scale comparison for better visualization of differences"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(data.sf1_queries))
    width = 0.35
    
    # SF-1 Log Scale
    bars1 = ax1.bar(x - width/2, data.sf1_gpu_times, width, label='GPU (Metal)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, data.sf1_duckdb_times, width, label='DuckDB', color='#3498db', alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms) - Log Scale', fontsize=12, fontweight='bold')
    ax1.set_title('SF-1 Dataset (1GB) - Log Scale Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, which='both')
    
    # SF-10 Log Scale
    bars3 = ax2.bar(x - width/2, data.sf10_gpu_times, width, label='GPU (Metal)', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, data.sf10_duckdb_times, width, label='DuckDB', color='#3498db', alpha=0.8)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (ms) - Log Scale', fontsize=12, fontweight='bold')
    ax2.set_title('SF-10 Dataset (10GB) - Log Scale Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data.sf10_queries, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'log_scale_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: log_scale_comparison.png")
    plt.close()

def plot_category_breakdown(data, output_dir):
    """Create breakdown by query category"""
    
    # Categorize queries
    categories = {
        'Selection': [0, 1, 2],
        'Aggregation': [3],
        'Join': [4],
        'TPC-H Queries': [5, 6, 7, 8, 9]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (cat_name, indices) in enumerate(categories.items()):
        ax = axes[idx]
        
        # Get queries for this category
        cat_queries = [data.sf1_queries[i] for i in indices]
        
        # SF-1 and SF-10 data
        sf1_gpu = [data.sf1_gpu_times[i] for i in indices]
        sf1_duckdb = [data.sf1_duckdb_times[i] for i in indices]
        sf10_gpu = [data.sf10_gpu_times[i] for i in indices]
        sf10_duckdb = [data.sf10_duckdb_times[i] for i in indices]
        
        x = np.arange(len(cat_queries))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, sf1_gpu, width, label='SF-1 GPU', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, sf1_duckdb, width, label='SF-1 DuckDB', color='#3498db', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, sf10_gpu, width, label='SF-10 GPU', color='#27ae60', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, sf10_duckdb, width, label='SF-10 DuckDB', color='#2980b9', alpha=0.8)
        
        ax.set_xlabel('Query', fontsize=11, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{cat_name} - Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_queries, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: category_breakdown.png")
    plt.close()

def plot_scaling_analysis(data, output_dir):
    """Analyze how performance scales from SF-1 to SF-10"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Calculate scaling factors (SF-10 / SF-1)
    gpu_scaling = [sf10 / sf1 for sf1, sf10 in zip(data.sf1_gpu_times, data.sf10_gpu_times)]
    duckdb_scaling = [sf10 / sf1 for sf1, sf10 in zip(data.sf1_duckdb_times, data.sf10_duckdb_times)]
    
    x = np.arange(len(data.sf1_queries))
    width = 0.35
    
    # Scaling Factor Comparison
    bars1 = ax1.bar(x - width/2, gpu_scaling, width, label='GPU (Metal)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, duckdb_scaling, width, label='DuckDB', color='#3498db', alpha=0.8)
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Expected (10x data)')
    
    ax1.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Scaling Factor (SF-10 / SF-1)', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Analysis: SF-10 vs SF-1', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}×', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}×', ha='center', va='bottom', fontsize=8)
    
    # Speedup Comparison: SF-1 vs SF-10
    bars3 = ax2.bar(x - width/2, data.sf1_speedup, width, label='SF-1', color='#e67e22', alpha=0.8)
    bars4 = ax2.bar(x + width/2, data.sf10_speedup, width, label='SF-10', color='#9b59b6', alpha=0.8)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    ax2.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GPU Speedup over DuckDB (×)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup: SF-1 vs SF-10', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data.sf10_queries, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scaling_analysis.png")
    plt.close()

def plot_performance_heatmap(data, output_dir):
    """Create a heatmap showing performance across queries and datasets"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Prepare data for heatmap (normalized)
    gpu_data = np.array([data.sf1_gpu_times, data.sf10_gpu_times])
    duckdb_data = np.array([data.sf1_duckdb_times, data.sf10_duckdb_times])
    
    # GPU Heatmap
    im1 = ax1.imshow(gpu_data, cmap='YlGn_r', aspect='auto')
    ax1.set_xticks(np.arange(len(data.sf1_queries)))
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    ax1.set_yticklabels(['SF-1 (1GB)', 'SF-10 (10GB)'])
    ax1.set_title('GPU (Metal) Execution Times (ms)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(data.sf1_queries)):
            text = ax1.text(j, i, f'{gpu_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Time (ms)')
    
    # DuckDB Heatmap
    im2 = ax2.imshow(duckdb_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(np.arange(len(data.sf1_queries)))
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(data.sf1_queries, rotation=45, ha='right')
    ax2.set_yticklabels(['SF-1 (1GB)', 'SF-10 (10GB)'])
    ax2.set_title('DuckDB Execution Times (ms)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(data.sf1_queries)):
            text = ax2.text(j, i, f'{duckdb_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Time (ms)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_heatmap.png")
    plt.close()

def generate_summary_report(data, output_dir):
    """Generate a text summary report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GPU (Metal) vs DuckDB Benchmark Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # SF-1 Summary
    report_lines.append("SF-1 Dataset (1GB) Summary:")
    report_lines.append("-" * 80)
    avg_sf1_speedup = np.mean(data.sf1_speedup)
    max_sf1_speedup = max(data.sf1_speedup)
    min_sf1_speedup = min(data.sf1_speedup)
    max_idx = data.sf1_speedup.index(max_sf1_speedup)
    min_idx = data.sf1_speedup.index(min_sf1_speedup)
    
    report_lines.append(f"Average GPU Speedup: {avg_sf1_speedup:.2f}×")
    report_lines.append(f"Maximum GPU Speedup: {max_sf1_speedup:.2f}× ({data.sf1_queries[max_idx].replace(chr(10), ' ')})")
    report_lines.append(f"Minimum GPU Speedup: {min_sf1_speedup:.2f}× ({data.sf1_queries[min_idx].replace(chr(10), ' ')})")
    report_lines.append("")
    
    # SF-10 Summary
    report_lines.append("SF-10 Dataset (10GB) Summary:")
    report_lines.append("-" * 80)
    avg_sf10_speedup = np.mean(data.sf10_speedup)
    max_sf10_speedup = max(data.sf10_speedup)
    min_sf10_speedup = min(data.sf10_speedup)
    max_idx = data.sf10_speedup.index(max_sf10_speedup)
    min_idx = data.sf10_speedup.index(min_sf10_speedup)
    
    report_lines.append(f"Average GPU Speedup: {avg_sf10_speedup:.2f}×")
    report_lines.append(f"Maximum GPU Speedup: {max_sf10_speedup:.2f}× ({data.sf10_queries[max_idx].replace(chr(10), ' ')})")
    report_lines.append(f"Minimum GPU Speedup: {min_sf10_speedup:.2f}× ({data.sf10_queries[min_idx].replace(chr(10), ' ')})")
    report_lines.append("")
    
    # Key Insights
    report_lines.append("Key Insights:")
    report_lines.append("-" * 80)
    
    # Find queries where GPU is faster
    gpu_faster_sf1 = sum(1 for s in data.sf1_speedup if s > 1)
    gpu_faster_sf10 = sum(1 for s in data.sf10_speedup if s > 1)
    
    report_lines.append(f"• GPU outperforms DuckDB in {gpu_faster_sf1}/10 queries for SF-1")
    report_lines.append(f"• GPU outperforms DuckDB in {gpu_faster_sf10}/10 queries for SF-10")
    report_lines.append("")
    
    # Performance categories
    report_lines.append("Performance Analysis by Category:")
    selection_speedup_sf1 = np.mean(data.sf1_speedup[:3])
    selection_speedup_sf10 = np.mean(data.sf10_speedup[:3])
    report_lines.append(f"• Selection queries: SF-1 avg {selection_speedup_sf1:.2f}×, SF-10 avg {selection_speedup_sf10:.2f}×")
    
    report_lines.append(f"• Aggregation: SF-1 {data.sf1_speedup[3]:.2f}×, SF-10 {data.sf10_speedup[3]:.2f}×")
    report_lines.append(f"• Join operations: SF-1 {data.sf1_speedup[4]:.2f}×, SF-10 {data.sf10_speedup[4]:.2f}×")
    
    tpch_speedup_sf1 = np.mean(data.sf1_speedup[5:])
    tpch_speedup_sf10 = np.mean(data.sf10_speedup[5:])
    report_lines.append(f"• TPC-H queries: SF-1 avg {tpch_speedup_sf1:.2f}×, SF-10 avg {tpch_speedup_sf10:.2f}×")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # Write to file
    report_path = output_dir / 'benchmark_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: benchmark_summary.txt")
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Benchmark Comparison Report Generator")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / 'benchmark_reports'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading benchmark data...")
    data = BenchmarkData()
    print("✓ Data loaded successfully")
    print()
    
    # Generate all visualizations
    print("Generating visualizations...")
    print()
    
    plot_execution_time_comparison(data, output_dir)
    plot_speedup_analysis(data, output_dir)
    plot_log_scale_comparison(data, output_dir)
    plot_category_breakdown(data, output_dir)
    plot_scaling_analysis(data, output_dir)
    plot_performance_heatmap(data, output_dir)
    
    print()
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(data, output_dir)
    
    print()
    print("=" * 80)
    print(f"✓ All reports generated successfully in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
