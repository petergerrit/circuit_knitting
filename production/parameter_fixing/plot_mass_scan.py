#!/usr/bin/env python3
"""
Plot mass scan results from run_mass_scan.py using Hank's style.
"""

import os
import json
import matplotlib.pyplot as plt

# Directories
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


def plot_mass_scan():
    """Generate mass scan plot using Hank's style."""
    # Load data from run_mass_scan.py output
    data = load_data("mass_scan_noiseless_eps0.05_shots1024.json")
    
    time_steps = data["time_steps"]
    masses = data["masses"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each mass
    mass_keys = {"m1p0": (1.0, hank_blue, "$m=1$"),
                 "m1p125": (1.125, hank_orange, "$m=1.125$"),
                 "m1p25": (1.25, hank_green, "$m=1.25$")}
    
    for mass_key, (mass_val, color, label) in mass_keys.items():
        fermion_numbers = data["data"][mass_key]["fermion_numbers"]
        
        # Hank's style: scatter + plot
        plt.scatter(time_steps, fermion_numbers, color=color, s=30)
        plt.plot(time_steps, fermion_numbers, color=color, label=label, alpha=0.7)
    
    plt.xlabel('evolution time')
    plt.ylabel('fermion number')
    plt.title(f'$\epsilon = 0.05$, noiseless, {data["num_shots"]} shots')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(PLOTS_DIR, 'mass_scan_noiseless.pdf')
    plt.savefig(output_file)
    plt.close()
    
    print(f"Plot saved to: {output_file}")


def main():
    """Generate the mass scan plot."""
    print("Generating mass scan plot with Hank's style...")
    plot_mass_scan()


if __name__ == "__main__":
    main()
