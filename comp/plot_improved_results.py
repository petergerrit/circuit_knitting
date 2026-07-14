#!/usr/bin/env python
"""Plot all ten improved/ results and show average with standard deviation."""

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
mean_be = np.mean(bootstrap_errors)
std_be = np.std(bootstrap_errors)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Individual fermion number results
run_indices = np.arange(1, num_runs + 1)
ax1.scatter(run_indices, fermion_numbers, color='blue', label='Individual runs', s=100)
ax1.axhline(mean_fn, color='red', linestyle='--', label=f'Average: {mean_fn:.6f}')
ax1.fill_between(run_indices, mean_fn - std_fn, mean_fn + std_fn, 
                 color='red', alpha=0.2, label=f'Std dev: {std_fn:.6f}')
ax1.set_xlabel('Run')
ax1.set_ylabel('Fermion Number')
ax1.set_title('Improved: Fermion Number per Run with Average ± Std Dev')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Individual bootstrap error results  
ax2.scatter(run_indices, bootstrap_errors, color='green', label='Individual runs', s=100)
ax2.axhline(mean_be, color='red', linestyle='--', label=f'Average: {mean_be:.6f}')
ax2.fill_between(run_indices, mean_be - std_be, mean_be + std_be,
                 color='red', alpha=0.2, label=f'Std dev: {std_be:.6f}')
ax2.set_xlabel('Run')
ax2.set_ylabel('Bootstrap Error')
ax2.set_title('Improved: Bootstrap Error per Run with Average ± Std Dev')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save and show
output_dir = "."
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "improved_step1_10runs_stats.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()

# Print summary statistics
print("\n=== Improved Results Summary ===")
print(f"Fermion Number: mean={mean_fn:.6f}, std={std_fn:.6f}")
print(f"Bootstrap Error: mean={mean_be:.6f}, std={std_be:.6f}")
