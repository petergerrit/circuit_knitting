#!/usr/bin/env python
# Plot the four noiseless cases with error bars

import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f]

shots_list = [1024 * 2**i for i in range(11)]

# Load all data
ferm_num_step1_no_noise_no_knit = load_data('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt')
bs_error_step1_no_noise_no_knit = load_data('data/bs_error_step1_epsilon08_no_noise_no_knit.txt')

ferm_num_step1_no_noise_knitted = load_data('data/ferm_num_step1_epsilon08_no_noise_knitted.txt')
bs_error_step1_no_noise_knitted = load_data('data/bs_error_step1_epsilon08_no_noise_knitted.txt')

ferm_num_step2_no_noise_no_knit = load_data('data/ferm_num_step2_epsilon08_no_noise_no_knit.txt')
bs_error_step2_no_noise_no_knit = load_data('data/bs_error_step2_epsilon08_no_noise_no_knit.txt')

ferm_num_step2_no_noise_knitted = load_data('data/ferm_num_step2_epsilon08_no_noise_knitted.txt')
bs_error_step2_no_noise_knitted = load_data('data/bs_error_step2_epsilon08_no_noise_knitted.txt')

# Plot all four cases
plt.figure(figsize=(12, 8))

# Step 1, no knit
plt.scatter(shots_list, ferm_num_step1_no_noise_no_knit, label='Step 1, no knit', color='blue')
plt.errorbar(shots_list, ferm_num_step1_no_noise_no_knit, bs_error_step1_no_noise_no_knit,
             alpha=0.2, ls='none', color='blue')

# Step 1, knitted
plt.scatter(shots_list, ferm_num_step1_no_noise_knitted, label='Step 1, knitted', color='blue', marker='s')
plt.errorbar(shots_list, ferm_num_step1_no_noise_knitted, bs_error_step1_no_noise_knitted,
             alpha=0.2, ls='none', color='blue')

# Step 2, no knit
plt.scatter(shots_list, ferm_num_step2_no_noise_no_knit, label='Step 2, no knit', color='orange')
plt.errorbar(shots_list, ferm_num_step2_no_noise_no_knit, bs_error_step2_no_noise_no_knit,
             alpha=0.2, ls='none', color='orange')

# Step 2, knitted
plt.scatter(shots_list, ferm_num_step2_no_noise_knitted, label='Step 2, knitted', color='orange', marker='s')
plt.errorbar(shots_list, ferm_num_step2_no_noise_knitted, bs_error_step2_no_noise_knitted,
             alpha=0.2, ls='none', color='orange')

plt.xscale('log')
plt.xlabel('shots')
plt.ylabel('mean fermion number')
plt.title('Noiseless Trotter Steps: Convergence with Shots')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig('figures/noiseless_convergence_comparison.pdf')
plt.show()

print("Plot saved to figures/noiseless_convergence_comparison.pdf")
