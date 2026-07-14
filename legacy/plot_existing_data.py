#!/usr/bin/env python
"""Plot all existing data from data/*.txt files."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Shot values: 1024 * 2^i for i in 0..10
shots_list = [1024 * 2**i for i in range(11)]

def load_data(filename):
    """Load data from a text file. Handles both 'value' and 'step->value' formats."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    values = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line has '->' delimiter
        if '->' in line:
            # Format: "step->value"
            parts = line.split('->')
            values.append(float(parts[-1].strip()))
        else:
            # Format: just the value
            values.append(float(line))
    
    return values

# Find all data files
data_dir = 'data'
pattern = r'^(ferm_num|bs_error)_step(\d+)_epsilon08_no_noise_(knitted|no_knit)\..*$'

print("Searching for data files...")

# Collect datasets
# Structure: {(step, variant): {'ferm_num': [...], 'bs_error': [...]}}
datasets = {}

for fname in sorted(os.listdir(data_dir)):
    match = re.match(pattern, fname)
    if not match:
        continue
    
    var_type = match.group(1)  # ferm_num or bs_error
    step = int(match.group(2))
    variant = match.group(3)   # knitted or no_knit
    
    key = (step, variant)
    if key not in datasets:
        datasets[key] = {}
    
    filepath = os.path.join(data_dir, fname)
    data = load_data(filepath)
    datasets[key][var_type] = data
    print(f"  Loaded: {fname} ({len(data)} points)")

print(f"\nFound {len(datasets)} dataset pairs")

# Color and marker configuration
colors = {
    1: 'blue',
    2: 'orange'
}
markers = {
    'no_knit': 'o',
    'knitted': 's'
}

plt.figure(figsize=(12, 8))

for (step, variant), data_dict in sorted(datasets.items()):
    if 'ferm_num' not in data_dict or 'bs_error' not in data_dict:
        continue
    
    ferm_num = data_dict['ferm_num']
    bs_error = data_dict['bs_error']
    
    # Use only the shots that match the data length
    n_points = len(ferm_num)
    shots = shots_list[:n_points]
    
    color = colors.get(step, 'gray')
    marker = markers.get(variant, 'x')
    label = f'Step {step}, {variant.replace("_", " ")}'
    
    plt.scatter(shots, ferm_num, label=label, 
                color=color, marker=marker, s=80)
    plt.errorbar(shots, ferm_num, bs_error, 
                 alpha=0.3, ls='none', color=color, capsize=3)

plt.xscale('log')
plt.xlabel('Shots (log scale)')
plt.ylabel('Mean Fermion Number')
plt.title('Noiseless Trotter Steps: Fermion Number vs Shots with Error Bars\n(Epsilon = 0.8)')
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig('figures/existing_data_convergence.pdf', dpi=300, bbox_inches='tight')
print("\nPlot saved to figures/existing_data_convergence.pdf")
plt.show()
