#!/usr/bin/env python
"""Temporary script to plot existing data."""

import numpy as np
import matplotlib.pyplot as plt

# Shot values
shots_list = [1024 * 2**i for i in range(11)]

# Load existing data
try:
    with open('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt', 'r') as f:
        ferm_num = [float(line.strip()) for line in f]
    with open('data/bs_error_step1_epsilon08_no_noise_no_knit.txt', 'r') as f:
        bs_error = [float(line.strip()) for line in f]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(shots_list[:len(ferm_num)], ferm_num, label='Step 1, no knit', color='blue')
    plt.errorbar(shots_list[:len(ferm_num)], ferm_num, bs_error[:len(ferm_num)], 
                 alpha=0.2, ls='none', color='blue')
    
    plt.xscale('log')
    plt.xlabel('shots')
    plt.ylabel('mean fermion number')
    plt.title('Existing Data: Step 1, No Noise, No Knit')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/existing_data.pdf')
    print("Plot saved to figures/existing_data.pdf")
    plt.show()
    
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
