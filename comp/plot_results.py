#!/usr/bin/env python
"""Plot improved results with all points at same x position for comparison."""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to improved results
improved_results_path = os.path.join("..", "improved", "results", "step1_knitted_10runs.json")

# Load results
with open(improved_results_path, 'r') as f:
    data = json.load(f)

# Extract fermion numbers and bootstrap errors
fermion_numbers = [r['fermion_number'] for r in data['results']]
bootstrap_errors = [r['bootstrap_error'] for r in data['results']]
num_runs = len(fermion_numbers)

# Calculate statistics
mean_fn = np.mean(fermion_numbers)
std_fn = np.std(fermion_numbers)

# Create figure with single panel
fig, ax = plt.subplots(figsize=(10, 8))

# X position for improved results (will be used for legacy comparison later)
x_improved = 1

# Plot individual points with error bars
# Use slight x offset for each point so they're visible but clustered
x_positions = np.ones(num_runs) * x_improved
# Add small jitter to avoid complete overlap
jitter = np.linspace(-0.1, 0.1, num_runs)
x_positions = x_positions + jitter

ax.scatter(x_positions, fermion_numbers, color='blue', label='Improved (individual)', s=100, alpha=0.6)

# Plot average with error bar showing std dev
ax.errorbar(x_improved, mean_fn, yerr=std_fn, 
            color='red', marker='o', markersize=12, capsize=5, 
            label=f'Improved: mean={mean_fn:.6f}, std={std_fn:.6f}')

ax.set_xlabel('Method')
ax.set_ylabel('Fermion Number')
ax.set_title('Step 1: Fermion Number Comparison (10 runs each)')
ax.legend()
ax.grid(True, alpha=0.3)

# Set x-axis ticks to accommodate future legacy points
# Assuming legacy will be at x=2
ax.set_xticks([x_improved, 2])
ax.set_xticklabels(['Improved', 'Legacy'])

plt.tight_layout()

# Save and show
output_dir = "."
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "step1_10_runs_stats.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()

# Print summary statistics
print("\n=== Improved Results Summary ===")
print(f"Fermion Number: mean={mean_fn:.6f}, std={std_fn:.6f}")
