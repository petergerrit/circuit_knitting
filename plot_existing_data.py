#!/usr/bin/env python
"""Plot existing data from the 4 available data files."""

import numpy as np
import matplotlib.pyplot as plt

# Shot values: 1024 * 2^i for i in 0..10
shots_list = [1024 * 2**i for i in range(11)]

def load_data(filename):
    """Load data from a text file."""
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f]

print("Loading existing data...")

# Load all existing data files
ferm_num_s1_nk = load_data('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt')
bs_error_s1_nk = load_data('data/bs_error_step1_epsilon08_no_noise_no_knit.txt')
ferm_num_s2_nk = load_data('data/ferm_num_step2_epsilon08_no_noise_no_knit.txt')
bs_error_s2_nk = load_data('data/bs_error_step2_epsilon08_no_noise_no_knit.txt')

print(f"Data loaded: {len(ferm_num_s1_nk)} points per dataset")

# Create plot
plt.figure(figsize=(12, 8))

# Step 1, no knit (blue, circle)
plt.scatter(shots_list, ferm_num_s1_nk, label='Step 1, no knit', color='blue', marker='o', s=80)
plt.errorbar(shots_list, ferm_num_s1_nk, bs_error_s1_nk, alpha=0.3, ls='none', color='blue', capsize=3)

# Step 2, no knit (orange, circle)
plt.scatter(shots_list, ferm_num_s2_nk, label='Step 2, no knit', color='orange', marker='o', s=80)
plt.errorbar(shots_list, ferm_num_s2_nk, bs_error_s2_nk, alpha=0.3, ls='none', color='orange', capsize=3)

plt.xscale('log')
plt.xlabel('Shots (log scale)')
plt.ylabel('Mean Fermion Number')
plt.title('Noiseless Trotter Steps: Fermion Number vs Shots\n(No Knitting, Epsilon = 0.8)')
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig('figures/existing_data_convergence.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/existing_data_convergence.png', dpi=300, bbox_inches='tight')
print("Plot saved to figures/existing_data_convergence.pdf and .png")
plt.show()
