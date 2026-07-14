#!/usr/bin/env python
"""Plot fermion number as a function of shots with error bars for all four cases."""

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Load data from a text file."""
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f]


# Shot values
shots_list = [1024 * 2**i for i in range(11)]

# Load all data
print("Loading data...")

# Step 1, no knit
ferm_num_s1_nk = load_data('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt')
bs_error_s1_nk = load_data('data/bs_error_step1_epsilon08_no_noise_no_knit.txt')

# Step 1, knitted
ferm_num_s1_k = load_data('data/ferm_num_step1_epsilon08_no_noise_knitted.txt')
bs_error_s1_k = load_data('data/bs_error_step1_epsilon08_no_noise_knitted.txt')

# Step 2, no knit
ferm_num_s2_nk = load_data('data/ferm_num_step2_epsilon08_no_noise_no_knit.txt')
bs_error_s2_nk = load_data('data/bs_error_step2_epsilon08_no_noise_no_knit.txt')

# Step 2, knitted
ferm_num_s2_k = load_data('data/ferm_num_step2_epsilon08_no_noise_knitted.txt')
bs_error_s2_k = load_data('data/bs_error_step2_epsilon08_no_noise_knitted.txt')

print("Data loaded successfully.")

# Create plot
plt.figure(figsize=(12, 8))

# Step 1, no knit (blue, circle)
plt.scatter(shots_list, ferm_num_s1_nk, label='Step 1, no knit', color='blue', marker='o')
plt.errorbar(shots_list, ferm_num_s1_nk, bs_error_s1_nk, alpha=0.2, ls='none', color='blue')

# Step 1, knitted (blue, square)
plt.scatter(shots_list, ferm_num_s1_k, label='Step 1, knitted', color='blue', marker='s')
plt.errorbar(shots_list, ferm_num_s1_k, bs_error_s1_k, alpha=0.2, ls='none', color='blue')

# Step 2, no knit (orange, circle)
plt.scatter(shots_list, ferm_num_s2_nk, label='Step 2, no knit', color='orange', marker='o')
plt.errorbar(shots_list, ferm_num_s2_nk, bs_error_s2_nk, alpha=0.2, ls='none', color='orange')

# Step 2, knitted (orange, square)
plt.scatter(shots_list, ferm_num_s2_k, label='Step 2, knitted', color='orange', marker='s')
plt.errorbar(shots_list, ferm_num_s2_k, bs_error_s2_k, alpha=0.2, ls='none', color='orange')

plt.xscale('log')
plt.xlabel('shots')
plt.ylabel('mean fermion number')
plt.title('Noiseless Trotter Steps: Fermion Number vs Shots with Error Bars')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig('figures/noiseless_convergence.pdf')
print("Plot saved to figures/noiseless_convergence.pdf")
plt.show()
