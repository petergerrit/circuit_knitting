#!/usr/bin/env python
"""Plot improved and legacy results for comparison."""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to results
improved_results_path = os.path.join("..", "improved", "results", "step1_knitted_10runs.json")
legacy_results_path = os.path.join("..", "legacy", "results", "step1_knitted_10runs.json")


# Load improved results
with open(improved_results_path, 'r') as f:
    improved_data = json.load(f)

# Extract improved fermion numbers and bootstrap errors
imp_fermion_numbers = [r['fermion_number'] for r in improved_data['results']]
imp_bootstrap_errors = [r['bootstrap_error'] for r in improved_data['results']]
num_runs = len(imp_fermion_numbers)

# Calculate improved statistics
imp_mean_fn = np.mean(imp_fermion_numbers)
imp_std_fn = np.std(imp_fermion_numbers)

# Load legacy results
with open(legacy_results_path, 'r') as f:
    legacy_data = json.load(f)

# Extract legacy fermion numbers and bootstrap errors
leg_fermion_numbers = [r['fermion_number'] for r in legacy_data['results']]
leg_bootstrap_errors = [r['bootstrap_error'] for r in legacy_data['results']]

# Calculate legacy statistics
leg_mean_fn = np.mean(leg_fermion_numbers)
leg_std_fn = np.std(leg_fermion_numbers)

# Create figure with single panel
fig, ax = plt.subplots(figsize=(10, 8))

# X positions
x_improved = 1
x_legacy = 2

# Plot improved individual points with error bars
x_imp_positions = np.ones(num_runs) * x_improved
jitter = np.linspace(-0.1, 0.1, num_runs)
x_imp_positions = x_imp_positions + jitter

ax.errorbar(x_imp_positions, imp_fermion_numbers, yerr=imp_bootstrap_errors, 
            color='blue', marker='o', markersize=8, capsize=3, 
            label='Improved (individual)', alpha=0.6, linestyle='none')

# Plot improved average band
band_width = 0.15
x_imp_band = np.linspace(x_improved - band_width, x_improved + band_width, 100)
ax.fill_between(x_imp_band, imp_mean_fn - imp_std_fn, imp_mean_fn + imp_std_fn,
                color='blue', alpha=0.2, label=f'Improved: mean={imp_mean_fn:.6f}, std={imp_std_fn:.6f}')
ax.hlines(imp_mean_fn, x_improved - band_width, x_improved + band_width,
          color='blue', linewidth=2, zorder=3)

# Plot legacy individual points with error bars
x_leg_positions = np.ones(num_runs) * x_legacy
x_leg_positions = x_leg_positions + jitter

ax.errorbar(x_leg_positions, leg_fermion_numbers, yerr=leg_bootstrap_errors, 
            color='orange', marker='s', markersize=8, capsize=3, 
            label='Legacy (individual)', alpha=0.6, linestyle='none')

# Plot legacy average band
x_leg_band = np.linspace(x_legacy - band_width, x_legacy + band_width, 100)
ax.fill_between(x_leg_band, leg_mean_fn - leg_std_fn, leg_mean_fn + leg_std_fn,
                color='orange', alpha=0.2, label=f'Legacy: mean={leg_mean_fn:.6f}, std={leg_std_fn:.6f}')
ax.hlines(leg_mean_fn, x_legacy - band_width, x_legacy + band_width,
          color='orange', linewidth=2, zorder=3)

ax.set_xlabel('Method')
ax.set_ylabel('Fermion Number')
ax.set_title('Step 1: Fermion Number Comparison (10 runs each)')
ax.legend()
ax.grid(True, alpha=0.3)

# Set x-axis ticks
ax.set_xticks([x_improved, x_legacy])
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
print("\n=== Results Summary ===")
print(f"Improved - Fermion Number: mean={imp_mean_fn:.6f}, std={imp_std_fn:.6f}")
print(f"Legacy   - Fermion Number: mean={leg_mean_fn:.6f}, std={leg_std_fn:.6f}")
