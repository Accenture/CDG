"""
Generate 1x4 histogram figure showing correct vs wrong answer distributions.
Figure (a): Mean Confidence (per trace)
Figure (b): Tail Confidence (per trace)
Figure (c): CDG Score (per answer)
Figure (d): Confidence Gradient (per trace)

This figure is designed for ICML paper format.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['lines.linewidth'] = 1.5

# Load data
json_path = '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp_histogram/cache/deepseek8b_aime2024_metrics.json'
with open(json_path, 'r') as f:
    data = json.load(f)

correct = data['correct']
wrong = data['wrong']
meta = data['meta']

# Create figure with 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(13, 2.5), dpi=300)

# Color scheme
color_correct = '#2ca02c'  # Green
color_wrong = '#d62728'    # Red

# Figure 1: Mean Confidence (per trace)
ax1 = axes[0]
ax1.hist(correct['mean_conf'], bins=50, alpha=0.6, color=color_correct,
         label='Correct', density=True, edgecolor='black', linewidth=0.5)
ax1.hist(wrong['mean_conf'], bins=50, alpha=0.6, color=color_wrong,
         label='Wrong', density=True, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('(a) Mean Token Confidence', fontweight='bold')
ax1.set_ylabel('Density', fontweight='bold')
ax1.set_title('Mean Confidence (per trace)', fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_ylim([0, None])  # Start from 0, auto max

# Figure 2: Tail Confidence (per trace)
ax2 = axes[1]
ax2.hist(correct['tail_conf'], bins=50, alpha=0.6, color=color_correct,
         label='Correct', density=True, edgecolor='black', linewidth=0.5)
ax2.hist(wrong['tail_conf'], bins=50, alpha=0.6, color=color_wrong,
         label='Wrong', density=True, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('(b) Tail Token Confidence', fontweight='bold')
ax2.set_ylabel('Density', fontweight='bold')
ax2.set_title('Tail Confidence (per trace)', fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_ylim([0, None])

# Figure 3: CDG Score (per answer)
ax3 = axes[2]
ax3.hist(correct['cdg_score'], bins=50, alpha=0.6, color=color_correct,
         label='Correct', density=True, edgecolor='black', linewidth=0.5)
ax3.hist(wrong['cdg_score'], bins=50, alpha=0.6, color=color_wrong,
         label='Wrong', density=True, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('(c) CDG Score', fontweight='bold')
ax3.set_ylabel('Density', fontweight='bold')
ax3.set_title('CDG Score (per answer)', fontweight='bold')
ax3.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.set_ylim([0, None])

# Figure 4: Gradient Direction Analysis - Multi-model comparison (bar chart)
ax4 = axes[3]

# Load data for all 4 models
models_data = {}
model_names = ['DeepSeek-8B', 'GPT-OSS-20B', 'Gemma-27B', 'QWQ-32B']
json_files = [
    '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp_histogram/cache/deepseek8b_aime2024_metrics.json',
    '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp_histogram/cache/gptoss20b_aime2024_metrics.json',
    '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp_histogram/cache/gemma3_27b_aime2024_metrics.json',
    '/home/azureuser/cloudfiles/code/Users/minghao.a.liu/sampling_credit_paper_results/results/exp_histogram/cache/qwq32b_aime2024_metrics.json'
]

for model_name, json_file in zip(model_names, json_files):
    with open(json_file, 'r') as f:
        models_data[model_name] = json.load(f)

# Calculate percentages for each model
pct_correct_positive = []
pct_wrong_negative = []

for model_name in model_names:
    correct_grad = np.array(models_data[model_name]['correct']['gradient'])
    wrong_grad = np.array(models_data[model_name]['wrong']['gradient'])

    pct_correct_positive.append((correct_grad > 0).sum() / len(correct_grad) * 100)
    pct_wrong_negative.append((wrong_grad < 0).sum() / len(wrong_grad) * 100)

# Create grouped bar chart
x = np.arange(2)  # 2 groups: correct positive, wrong negative
width = 0.18  # Width of each bar
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

# Model colors (using distinct colors)
model_colors = ['#9467bd', '#e377c2', '#17becf', '#bcbd22']  # Purple, Pink, Cyan, Yellow-green

# Plot bars for each model
for i, model_name in enumerate(model_names):
    values = [pct_correct_positive[i], pct_wrong_negative[i]]
    bars = ax4.bar(x + offsets[i], values, width, label=model_name,
                   color=model_colors[i], alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add percentage labels on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}',
                 ha='center', va='bottom', fontsize=6, fontweight='bold')

ax4.set_xlabel(r'(d) Direction of $\nabla C_{\ell}$', fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontweight='bold')
ax4.set_title('Gradient Analysis (4 Models)', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([r'Correct (positive $\nabla C_{\ell}$)', r'Wrong (negative $\nabla C_{\ell}$)'])
ax4.set_ylim([0, 115])  # Leave more room for legend and labels
ax4.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=5,
          handlelength=1.0, handletextpad=0.3, borderpad=0.2, labelspacing=0.15)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Adjust layout to reduce space between figures
plt.tight_layout(pad=0.5, w_pad=1.0)

# Save figure
output_path = '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/histogram_figure.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to: {output_path}")

# Also save as PNG for preview
output_path_png = '/home/azureuser/cloudfiles/code/Users/yu.bu.wang/sampling_credit/scripts/histogram_figure.png'
plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300)
print(f"PNG preview saved to: {output_path_png}")

# Print summary
print(f"\nSummary:")
print(f"Model: DeepSeek R1 8B")
print(f"Dataset: AIME 2024")
print(f"Questions: {meta['n_questions']}")
print(f"Correct answers: {meta['n_correct_answers']}")
print(f"Wrong answers: {meta['n_wrong_answers']}")

plt.show()
