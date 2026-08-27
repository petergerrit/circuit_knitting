#!/usr/bin/env python3
"""
Plot results from parameter fixing experiments.

Produces mass_scan.pdf and trotter_vs_exact.pdf figures using Hank's style.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Hank's color scheme
hank_blue = (0, 114/256, 178/256)
hank_orange = (213/256, 94/256, 0)
hank_green = (0, 178/256, 114/256)


def load_aggregated_data(filename):
    """Load aggregated data from JSON file."""
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        return json.load(f)


def plot_mass_scan():
    """Generate mass_scan.pdf figure using Hank's style."""
    # Load aggregated data
    agg_data = load_aggregated_data("agg_mass_scan_eps0p005.json")
    
    full_steps = agg_data["steps"]
    data = agg_data["data"]
    
    ferm_num_m1p0 = data["m1p0"]["fermion_number"]
    bs_error_m1p0 = data["m1p0"]["bootstrap_error"]
    ferm_num_m1p125 = data["m1p125"]["fermion_number"]
    bs_error_m1p125 = data["m1p125"]["bootstrap_error"]
    ferm_num_m1p25 = data["m1p25"]["fermion_number"]
    bs_error_m1p25 = data["m1p25"]["bootstrap_error"]

    plt.figure()
    
    # Hank's style: scatter + errorbar + plot
    plt.scatter(full_steps, ferm_num_m1p0, color=hank_blue)
    plt.errorbar(full_steps, ferm_num_m1p0, bs_error_m1p0, alpha=0.2, ls='none', color=hank_blue)
    plt.plot(full_steps, ferm_num_m1p0, color=hank_blue, label='fit: $m=1$')

    plt.scatter(full_steps, ferm_num_m1p125, color=hank_orange)
    plt.errorbar(full_steps, ferm_num_m1p125, bs_error_m1p125, alpha=0.2, ls='none', color=hank_orange)
    plt.plot(full_steps, ferm_num_m1p125, color=hank_orange, label='fit: $m=1.125$')

    plt.scatter(full_steps, ferm_num_m1p25, color=hank_green)
    plt.errorbar(full_steps, ferm_num_m1p25, bs_error_m1p25, alpha=0.2, ls='none', color=hank_green)
    plt.plot(full_steps, ferm_num_m1p25, color=hank_green, label='fit: $m=1.25$')

    plt.xlabel('evolution time')
    plt.ylabel('mean fermion number')
    plt.title('$\\epsilon=0.05$')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'mass_scan.pdf'))
    plt.close()


def plot_trotter_vs_exact():
    """Generate trotter_vs_exact.pdf figure using Hank's style."""
    # Load aggregated data
    agg_data = load_aggregated_data("agg_trotter_vs_exact_m1p125.json")
    
    short_steps = agg_data["steps"]
    data = agg_data["data"]
    
    ferm_num_m1p125_e0p005 = data["exact"]["fermion_number"]
    bs_error_m1p125_e0p005 = data["exact"]["bootstrap_error"]
    ferm_num_m1p125_trot = data["trotter"]["fermion_number"]
    bs_error_m1p125_trot = data["trotter"]["bootstrap_error"]

    plt.figure()

    # Hank's style for trotter_vs_exact
    plt.scatter(short_steps, ferm_num_m1p125_e0p005[:9], color=hank_blue)
    plt.plot(short_steps, ferm_num_m1p125_e0p005[:9], color=hank_blue, label='fit: truncated ($\\epsilon=0.05$)')
    plt.errorbar(short_steps, ferm_num_m1p125_e0p005[:9], bs_error_m1p125_e0p005[:9], alpha=0.2, ls='none', color=hank_blue)

    plt.plot(short_steps, ferm_num_m1p125_trot, color=hank_orange, label='fit: Trotterized')
    plt.scatter(short_steps[0], ferm_num_m1p125_trot[0], color=hank_orange, label='step 0')
    plt.scatter(short_steps[1:5], ferm_num_m1p125_trot[1:5], color=hank_orange, label='step 1', marker='s')
    plt.scatter(short_steps[5:], ferm_num_m1p125_trot[5:], color=hank_orange, label='step 2', marker='*')
    plt.errorbar(short_steps, ferm_num_m1p125_trot, bs_error_m1p125_trot, alpha=0.2, ls='none', color=hank_orange)

    plt.xlabel('evolution time')
    plt.ylabel('mean fermion number')
    plt.title('$m=1.125$')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'trotter_vs_exact.pdf'))
    plt.close()


def main():
    """Generate all plots from experiment data."""
    print("Generating plots with Hank's style...")
    plot_mass_scan()
    plot_trotter_vs_exact()
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
