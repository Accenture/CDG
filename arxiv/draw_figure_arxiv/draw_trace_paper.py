"""
Generate 1x4 figure showing scaling trends and beta ablation.
Figure 1: DeepSeek-R1-8B on AIME 2025 (trace scaling)
Figure 2: Gemma-3-27B on AIME 2024 (trace scaling)
Figure 3: QWQ-32B on BRUMO 2025 (trace scaling)
Figure 4: DeepSeek-8B beta ablation (alpha=0.5)

This figure is designed for a cross-column ICML paper (wide, rectangular format).
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 6.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5

# Data
trace_counts = [8, 16, 32, 64, 128, 256, 512]

# DeepSeek-R1-8B on AIME 2025
deepseek_majority = [81.3, 83.3, 82.7, 83.3, 83.3, 83.3, 83.3]
deepseek_mean = [81.3, 83.3, 83.3, 83.3, 83.3, 83.3, 83.3]
deepseek_top10 = [79.3, 84.0, 84.0, 84.7, 84.0, 84.0, 83.3]
deepseek_cdg = [82.0, 86.0, 86.0, 87.3, 87.3, 88.0, 93.3]

# Gemma-3-27B on AIME 2024
gemma_majority = [46.7, 43.3, 47.3, 46.7, 49.3, 48.0, 50.0]
gemma_mean = [48.7, 43.3, 48.0, 48.0, 50.0, 49.3, 50.0]
gemma_top10 = [40.7, 44.0, 50.0, 47.3, 51.3, 53.3, 53.3]
gemma_cdg = [50.7, 48.7, 53.3, 52.7, 52.7, 56.0, 56.7]

# QWQ-32B on BRUMO 2025
qwq_majority = [80.7, 81.3, 80.7, 81.3, 80.7, 80.0, 80.0]
qwq_mean = [81.3, 82.7, 80.7, 81.3, 80.7, 80.0, 80.0]
qwq_top10 = [82.0, 80.0, 82.7, 83.3, 84.0, 84.7, 86.7]
qwq_cdg = [84.7, 84.7, 86.7, 88.0, 88.0, 90.0, 90.0]

# DeepSeek-8B Beta Ablation (alpha=0.5, position_pct=10%)
# Extract data from JSON file
import json
json_path = '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp3_cdg_sweep/cdg_sweep_results_20260120_045401.json'
with open(json_path, 'r') as f:
    sweep_data = json.load(f)

# Filter for deepseek8b with alpha=0.5, position_pct=10
deepseek_sweep = [
    entry for entry in sweep_data['results']
    if entry['model'] == 'deepseek8b' and entry['alpha'] == 0.5 and entry['position_pct'] == 10
]

# Organize by dataset
beta_values = sorted(set(entry['beta'] for entry in deepseek_sweep))
datasets_beta = {
    'aime2024': [],
    'aime2025': [],
    'bruno2025': [],
    'hmmt2025': []
}

for dataset in datasets_beta.keys():
    for beta in beta_values:
        entry = [e for e in deepseek_sweep if e['dataset'] == dataset and e['beta'] == beta][0]
        datasets_beta[dataset].append(entry['accuracy'] * 100)  # Convert to percentage

# Create figure with 1 row, 4 columns
# Use wider aspect ratio for cross-column ICML format
fig, axes = plt.subplots(1, 4, figsize=(13, 2.5), dpi=300)

# Color scheme - professional colors
colors = {
    'Majority': '#1f77b4',        # Blue
    'DeepConf-Mean': '#ff7f0e',   # Orange
    'DeepConf-Tail': '#2ca02c',   # Green
    'CDG': '#d62728'              # Red (highlight)
}

markers = {
    'Majority': 'o',
    'DeepConf-Mean': 's',
    'DeepConf-Tail': '^',
    'CDG': 'D'
}

# Figure 1: DeepSeek-R1-8B on AIME 2025
ax1 = axes[0]
ax1.plot(trace_counts, deepseek_majority, label='Majority',
         color=colors['Majority'], marker=markers['Majority'], linewidth=1.5)
ax1.plot(trace_counts, deepseek_mean, label='DeepConf-Mean',
         color=colors['DeepConf-Mean'], marker=markers['DeepConf-Mean'], linewidth=1.5)
ax1.plot(trace_counts, deepseek_top10, label='DeepConf-Tail',
         color=colors['DeepConf-Tail'], marker=markers['DeepConf-Tail'], linewidth=1.5)
ax1.plot(trace_counts, deepseek_cdg, label='CDG (ours)',
         color=colors['CDG'], marker=markers['CDG'], linewidth=2.0)

ax1.set_xlabel(r'(a) Number of Traces $L$', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('DeepSeek-R1-8B on AIME 2025', fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.set_xticks(trace_counts)
ax1.set_xticklabels([str(t) for t in trace_counts])
ax1.set_ylim([78, 95])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
          handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)

# Figure 2: Gemma-3-27B on AIME 2024
ax2 = axes[1]
ax2.plot(trace_counts, gemma_majority, label='Majority',
         color=colors['Majority'], marker=markers['Majority'], linewidth=1.5)
ax2.plot(trace_counts, gemma_mean, label='DeepConf-Mean',
         color=colors['DeepConf-Mean'], marker=markers['DeepConf-Mean'], linewidth=1.5)
ax2.plot(trace_counts, gemma_top10, label='DeepConf-Tail',
         color=colors['DeepConf-Tail'], marker=markers['DeepConf-Tail'], linewidth=1.5)
ax2.plot(trace_counts, gemma_cdg, label='CDG (ours)',
         color=colors['CDG'], marker=markers['CDG'], linewidth=2.0)

ax2.set_xlabel(r'(b) Number of Traces $L$', fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title('Gemma-3-27B on AIME 2024', fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.set_xticks(trace_counts)
ax2.set_xticklabels([str(t) for t in trace_counts])
ax2.set_ylim([38, 62])
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
          handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)

# Figure 3: QWQ-32B on BRUNO 2025
ax3 = axes[2]
ax3.plot(trace_counts, qwq_majority, label='Majority',
         color=colors['Majority'], marker=markers['Majority'], linewidth=1.5)
ax3.plot(trace_counts, qwq_mean, label='DeepConf-Mean',
         color=colors['DeepConf-Mean'], marker=markers['DeepConf-Mean'], linewidth=1.5)
ax3.plot(trace_counts, qwq_top10, label='DeepConf-Tail',
         color=colors['DeepConf-Tail'], marker=markers['DeepConf-Tail'], linewidth=1.5)
ax3.plot(trace_counts, qwq_cdg, label='CDG (ours)',
         color=colors['CDG'], marker=markers['CDG'], linewidth=2.0)

ax3.set_xlabel(r'(c) Number of Traces $L$', fontweight='bold')
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_title('QWQ-32B on BRUMO 2025', fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.set_xticks(trace_counts)
ax3.set_xticklabels([str(t) for t in trace_counts])
ax3.set_ylim([78, 92])
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.legend(loc='upper left', framealpha=0.9, edgecolor='gray',
          handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)

# Figure 4: DeepSeek-8B Beta Ablation
ax4 = axes[3]

# Colors for each dataset (distinct from method colors in other subfigures)
dataset_colors = {
    'aime2024': '#9467bd',    # Purple
    'aime2025': '#e377c2',    # Pink
    'bruno2025': '#17becf',   # Cyan
    'hmmt2025': '#bcbd22'     # Yellow-green
}

# Dataset display names (capitalized for legend)
dataset_labels = {
    'aime2024': 'AIME2024',
    'aime2025': 'AIME2025',
    'bruno2025': 'BRUMO2025',
    'hmmt2025': 'HMMT2025'
}

# Plot each dataset
for dataset, accuracies in datasets_beta.items():
    ax4.plot(beta_values, accuracies,
             color=dataset_colors[dataset], marker='o', linewidth=1.5, markersize=5,
             label=dataset_labels[dataset])

# Add a vertical line at beta=10 (the optimal value used in paper)
ax4.axvline(x=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

ax4.set_xlabel(r'(d) CDG weight $\beta$', fontweight='bold')
ax4.set_ylabel('Accuracy (%)', fontweight='bold')
ax4.set_title(r'DeepSeek-R1-8B ($\alpha$=0.5)', fontweight='bold')
ax4.set_xticks(beta_values)
ax4.set_xticklabels([str(b) for b in beta_values])
ax4.set_ylim([60, 105])
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax4.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
          handlelength=1.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.3)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save figure
output_path = '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/trace_scaling_figure.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to: {output_path}")

# Also save as PNG for preview
output_path_png = '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/trace_scaling_figure.png'
plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300)
print(f"PNG preview saved to: {output_path_png}")

plt.show()
