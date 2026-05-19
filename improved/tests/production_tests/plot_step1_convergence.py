#!/usr/bin/env python3
"""
Plot fermion number convergence as a function of shots for step 1 knitted results.

Loads results from step1_knitted_all_shots.json and plots fermion number
with bootstrap error bars vs. number of shots.
Also adds fermion number with bootstrap error band from step1_summary_seed42.json.
"""

import sys
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

results_dir = "tests/production_tests/results"
input_file = os.path.join(results_dir, "step1_knitted_all_shots.json")
summary_file = os.path.join(results_dir, "step1_summary_seed42.json")
output_file = os.path.join(results_dir, "step1_convergence_noiseless.pdf")

# Load the combined results
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract shot counts, fermion numbers, and errors
shots = [r['num_shots'] for r in data['results']]
fermion_numbers = [r['fermion_number'] for r in data['results']]
bootstrap_errors = [r['bootstrap_error'] for r in data['results']]

# Load summary data for error band
with open(summary_file, 'r') as f:
    summary = json.load(f)

summary_fermion = summary['fermion_number']
summary_error = summary['bootstrap_error']

# Get data range for x-axis
x_min = min(shots)
x_max = max(shots)

# Create plot with A4 aspect ratio (1:√2)
plt.figure(figsize=(14.14, 10))
plt.errorbar(shots, fermion_numbers, yerr=bootstrap_errors, 
             fmt='o', capsize=5, markersize=8, color='blue', ecolor='darkblue',
             label='Knitted results')

plt.xscale('log')

# Set x-axis limits with more padding so error bars fit
plt.xlim(x_min / 1.20, x_max * 1.20)

# Add error band across entire plot width (edge to edge)
# Use the actual axis limits to extend to edges
ax = plt.gca()
band_x_min, band_x_max = ax.get_xlim()
plt.fill_between([band_x_min, band_x_max], 
                 [summary_fermion - summary_error, summary_fermion - summary_error],
                 [summary_fermion + summary_error, summary_fermion + summary_error],
                 color='red', alpha=0.2)

# Add horizontal line for summary fermion number
plt.axhline(y=summary_fermion, color='red', linestyle='--', linewidth=2)
plt.xlabel('Number of Shots', fontsize=24)
plt.ylabel('Fermion Number', fontsize=24)
plt.title('Step 1 Fermion Number Convergence (Noiseless)', fontsize=28)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Create custom legend with combined entry for summary
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend_handler as hl

class SummaryHandler(hl.HandlerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Center the line and band vertically on the text baseline
        ycenter = ydescent + height / 2
        # Create a filled rectangle with double height, centered on ycenter
        fill = Rectangle((xdescent, ycenter - height), width, height * 2,
                        facecolor='red', alpha=0.2, edgecolor='none',
                        transform=trans)
        # Create a dashed line through the middle at ycenter
        line = Line2D([xdescent, xdescent + width], [ycenter, ycenter],
                     color='red', linestyle='--', linewidth=2, transform=trans)
        return [fill, line]

class KnittedHandler(hl.HandlerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Center the marker and error bar vertically on the text baseline
        ycenter = ydescent + height / 2
        # Create marker at center
        marker = Line2D([xdescent + width/2], [ycenter],
                       marker='o', color='w', markerfacecolor='blue',
                       markeredgecolor='darkblue', markersize=8,
                       transform=trans)
        # Create error bar (vertical line)
        capsize = 8
        err_line = Line2D([xdescent + width/2, xdescent + width/2],
                          [ycenter - capsize, ycenter + capsize],
                          color='darkblue', linewidth=1, transform=trans)
        # Horizontal caps
        cap_half = capsize / 2
        cap1 = Line2D([xdescent + width/2 - cap_half, xdescent + width/2 + cap_half],
                     [ycenter - capsize, ycenter - capsize],
                     color='darkblue', linewidth=1, transform=trans)
        cap2 = Line2D([xdescent + width/2 - cap_half, xdescent + width/2 + cap_half],
                     [ycenter + capsize, ycenter + capsize],
                     color='darkblue', linewidth=1, transform=trans)
        return [marker, err_line, cap1, cap2]

# Create proxy artists
knitted_proxy = Rectangle((0, 0), 1, 1, visible=False)
summary_proxy = Line2D([0], [0], color='red', linestyle='--', linewidth=2, visible=False)

plt.legend([knitted_proxy, summary_proxy],
           [f'knitted results', f'non-knitted result\n({summary["num_shots"]} shots)'],
           handler_map={Rectangle: KnittedHandler(), Line2D: SummaryHandler()},
           fontsize=20)

plt.tight_layout()

# Save as PDF
plt.savefig(output_file, format='pdf', bbox_inches='tight')
plt.close()

print(f"Plot saved to {output_file}")
