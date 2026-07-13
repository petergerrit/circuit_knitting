#!/usr/bin/env python
"""Plot existing data from the 4 available data files."""

import numpy as np
import matplotlib.pyplot as plt

# Shot values: 1024 * 2^i for i in 0..10
shots_list_full = [1024 * 2**i for i in range(11)]

def load_data(filename):
    """Load data from a text file, return empty list if file doesn't exist."""
    try:
        with open(filename, 'r') as f:
            return [float(line.strip()) for line in f]
    except FileNotFoundError:
        return []

print("Loading existing data...")

# Load all existing data files (no-knit)
ferm_num_s1_nk = load_data('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt')
bs_error_s1_nk = load_data('data/bs_error_step1_epsilon08_no_noise_no_knit.txt')
ferm_num_s2_nk = load_data('data/ferm_num_step2_epsilon08_no_noise_no_knit.txt')
bs_error_s2_nk = load_data('data/bs_error_step2_epsilon08_no_noise_no_knit.txt')

# Load knitted data if available
ferm_num_s1_k = load_data('data/ferm_num_step1_epsilon08_no_noise_knitted.txt')
bs_error_s1_k = load_data('data/bs_error_step1_epsilon08_no_noise_knitted.txt')
ferm_num_s2_k = load_data('data/ferm_num_step2_epsilon08_no_noise_knitted.txt')
bs_error_s2_k = load_data('data/bs_error_step2_epsilon08_no_noise_knitted.txt')

# Determine which datasets are available
no_knit_available = all([ferm_num_s1_nk, bs_error_s1_nk, ferm_num_s2_nk, bs_error_s2_nk])
knitted_available = all([ferm_num_s1_k, bs_error_s1_k, ferm_num_s2_k, bs_error_s2_k])

# Determine the number of complete data points (minimum across all available arrays)
all_data = [ferm_num_s1_nk, bs_error_s1_nk, ferm_num_s2_nk, bs_error_s2_nk,
           ferm_num_s1_k, bs_error_s1_k, ferm_num_s2_k, bs_error_s2_k]
loaded_data = [d for d in all_data if d]  # Only include non-empty datasets
num_points = min(len(d) for d in loaded_data) if loaded_data else 0
shots_list = shots_list_full[:num_points]

print(f"Data loaded: {num_points} points per dataset (using first {num_points} of {len(shots_list_full)} possible shots)")
if no_knit_available:
    print("  No-knit data: available")
else:
    print("  No-knit data: NOT available")
if knitted_available:
    print("  Knitted data: available")
else:
    print("  Knitted data: NOT available")

# Create plot
plt.figure(figsize=(12, 8))

# Step 1, no knit (blue, circle)
if no_knit_available:
    plt.scatter(shots_list, ferm_num_s1_nk[:num_points], label='Step 1, no knit', color='blue', marker='o', s=80)
    plt.errorbar(shots_list, ferm_num_s1_nk[:num_points], bs_error_s1_nk[:num_points], alpha=0.3, ls='none', color='blue', capsize=3)

# Step 1, knitted (blue, square)
if knitted_available:
    plt.scatter(shots_list, ferm_num_s1_k[:num_points], label='Step 1, knitted', color='blue', marker='s', s=80)
    plt.errorbar(shots_list, ferm_num_s1_k[:num_points], bs_error_s1_k[:num_points], alpha=0.3, ls='none', color='blue', capsize=3)

# Step 2, no knit (orange, circle)
if no_knit_available:
    plt.scatter(shots_list, ferm_num_s2_nk[:num_points], label='Step 2, no knit', color='orange', marker='o', s=80)
    plt.errorbar(shots_list, ferm_num_s2_nk[:num_points], bs_error_s2_nk[:num_points], alpha=0.3, ls='none', color='orange', capsize=3)

# Step 2, knitted (orange, square)
if knitted_available:
    plt.scatter(shots_list, ferm_num_s2_k[:num_points], label='Step 2, knitted', color='orange', marker='s', s=80)
    plt.errorbar(shots_list, ferm_num_s2_k[:num_points], bs_error_s2_k[:num_points], alpha=0.3, ls='none', color='orange', capsize=3)

plt.xscale('log')
plt.xlabel('Shots (log scale)')
plt.ylabel('Mean Fermion Number')
plt.title('Noiseless Trotter Steps: Fermion Number vs Shots\n(Epsilon = 0.8)')
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig('figures/existing_data_convergence.pdf', dpi=300, bbox_inches='tight')
print("Plot saved to figures/existing_data_convergence.pdf")
plt.show()
