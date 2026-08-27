#!/usr/bin/env python3
"""
Plot results from trotter comparison experiment.

Produces trotter_comp.pdf using Hank's style.
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


def load_data(filename):
    """Load data from JSON file."""
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        return json.load(f)


def main():
    """Generate trotter_comp.pdf figure."""
    # Load data
    data = load_data("trotter_comp.json")
    
    # Extract Curve 1 data (9 points)
    curve1_points = data["curve1"]["points"]
    times1 = [p["time"] for p in curve1_points]
    ferm_nums1 = [p["fermion_number"] for p in curve1_points]
    boot_errors1 = [p["bootstrap_error"] for p in curve1_points]
    
    # Extract Curve 2 data (9 points) - need to sort by time
    curve2_points = sorted(data["curve2"]["points"], key=lambda p: p["time"])
    times2 = [p["time"] for p in curve2_points]
    ferm_nums2 = [p["fermion_number"] for p in curve2_points]
    boot_errors2 = [p["bootstrap_error"] for p in curve2_points]
    
    plt.figure(figsize=(10, 6))
    
    # Hank's style: scatter + errorbar + plot for Curve 1 (blue)
    plt.scatter(times1, ferm_nums1, color=hank_blue)
    plt.errorbar(times1, ferm_nums1, boot_errors1, alpha=0.2, ls='none', color=hank_blue)
    plt.plot(times1, ferm_nums1, color=hank_blue, 
             label=rf'Curve 1: $\epsilon=0.05$, trotter_steps=[0,4,...,32]')
    
    # Hank's style: scatter + errorbar + plot for Curve 2 (orange)
    plt.scatter(times2, ferm_nums2, color=hank_orange)
    plt.errorbar(times2, ferm_nums2, boot_errors2, alpha=0.2, ls='none', color=hank_orange)
    plt.plot(times2, ferm_nums2, color=hank_orange, 
             label=r'Curve 2: mixed trotter steps and $\epsilon$')
    
    plt.xlabel('time')
    plt.ylabel('mean fermion number')
    plt.title('Trotter Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'trotter_comp.pdf'))
    plt.close()
    
    print(f"Plot saved to: {os.path.join(PLOTS_DIR, 'trotter_comp.pdf')}")


if __name__ == "__main__":
    main()
