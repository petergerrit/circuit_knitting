#!/usr/bin/env python3
"""
Create fermion_number_with_vs_without_knitting_scatter.pdf plot
using data from the current directory structure.

Matches the style of ~/git/circuit_knitting/plots.ipynb
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set font conventions to match the notebook
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

# Colors matching the original notebook (swapped for knitted/non-knitted)
hank_blue = (213/256, 94/256, 0)  # orange - now used for no knitting
hank_orange = (0, 114/256, 178/256)  # blue - now used for knitting

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'fermion_number_with_vs_without_knitting_scatter.pdf')


def load_json_files(directory):
    """Load all JSON files from a directory."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data = []
    for f in sorted(json_files):
        with open(f, 'r') as file:
            data.append(json.load(file))
    return data


def get_time_from_file(file_data):
    """Compute time = trotter_step * epsilon from file metadata."""
    return file_data['trotter_step'] * file_data['epsilon']


def main():
    # Set up figure
    fig, ax = plt.subplots()
    
    # Load resampled arrays
    resampling_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'resampling')
    trotter_steps = np.load(os.path.join(resampling_dir, 'trotter_steps.npy'))
    knitted_fermion = np.load(os.path.join(resampling_dir, 'knitted_resampled_fermion_number.npy'))
    knitted_std = np.load(os.path.join(resampling_dir, 'knitted_resampled_std.npy'))
    no_knit_fermion = np.load(os.path.join(resampling_dir, 'no_knit_resampled_fermion_number.npy'))
    no_knit_std = np.load(os.path.join(resampling_dir, 'no_knit_resampled_std.npy'))
    
    # Load exact data for continuous black line
    exact_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'exact')
    ferm_num_exact = np.load(os.path.join(exact_dir, 'ferm_num_exact_no_noise_no_knit.npy'))
    # Trim exact data to match trotter_steps length (as done in notebook line 211)
    ferm_num_exact = ferm_num_exact[:len(trotter_steps)]
    
    # Plot exact line (continuous black line)
    ax.plot(trotter_steps, ferm_num_exact, color='black', label='exact')
    
    # Load all JSON data for individual points
    knitting_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'knitting', 'results')
    no_knitting_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'no_knitting', 'results')
    
    knitting_data = load_json_files(knitting_dir)
    no_knitting_data = load_json_files(no_knitting_dir)
    
    # Group knitting data by time
    knitting_by_time = {}
    for file_data in knitting_data:
        time = get_time_from_file(file_data)
        if time not in knitting_by_time:
            knitting_by_time[time] = []
        for run in file_data['results']:
            knitting_by_time[time].append({
                'fermion_number': run['fermion_number'],
                'bootstrap_error': run['bootstrap_error']
            })
    
    # Group no_knitting data by time
    no_knitting_by_time = {}
    for file_data in no_knitting_data:
        time = get_time_from_file(file_data)
        if time not in no_knitting_by_time:
            no_knitting_by_time[time] = []
        for run in file_data['results']:
            no_knitting_by_time[time].append({
                'fermion_number': run['fermion_number'],
                'bootstrap_error': run['bootstrap_error']
            })
    
    # Sort time points
    all_times = sorted(set(list(knitting_by_time.keys()) + list(no_knitting_by_time.keys())))
    
    # Plot resampled fill_between bands
    # Filter out NaN values for plotting
    valid_knit_idx = ~np.isnan(knitted_fermion)
    valid_no_knit_idx = ~np.isnan(no_knit_fermion)
    
    # No knitting band (blue)
    ax.fill_between(
        trotter_steps[valid_no_knit_idx],
        no_knit_fermion[valid_no_knit_idx] + no_knit_std[valid_no_knit_idx],
        no_knit_fermion[valid_no_knit_idx] - no_knit_std[valid_no_knit_idx],
        color=hank_blue, alpha=0.2
    )
    
    # Knitting band (orange)
    ax.fill_between(
        trotter_steps[valid_knit_idx],
        knitted_fermion[valid_knit_idx] + knitted_std[valid_knit_idx],
        knitted_fermion[valid_knit_idx] - knitted_std[valid_knit_idx],
        color=hank_orange, alpha=0.2
    )
    
    # Plot individual scatter points for no_knitting (no labels - will use custom legend)
    for time in all_times:
        if time in no_knitting_by_time:
            for run in no_knitting_by_time[time]:
                ax.scatter(time, run['fermion_number'], color=hank_blue)
                ax.errorbar(time, run['fermion_number'], run['bootstrap_error'], 
                           ls='none', color=hank_blue, elinewidth=0.5)
    
    # Plot individual scatter points for knitting (no labels - will use custom legend)
    for time in all_times:
        if time in knitting_by_time:
            for run in knitting_by_time[time]:
                ax.scatter(time, run['fermion_number'], color=hank_orange)
                ax.errorbar(time, run['fermion_number'], run['bootstrap_error'], 
                           ls='none', color=hank_orange, elinewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('evolution time')
    ax.set_ylabel('mean fermion number')
    ax.set_title('Time Evolution of Mean Fermion Number')
    
    # Legend - custom handles for 5 entries: 2 bands + exact line + 2 dots
    import matplotlib.patches as mpatches
    no_knit_band = mpatches.Patch(color=hank_blue, alpha=0.2, label='resampling (no knitting)')
    knit_band = mpatches.Patch(color=hank_orange, alpha=0.2, label='resampling (with knitting)')
    exact_line = Line2D([0], [0], label='exact', color='black', linestyle='-')
    no_knit_dot = Line2D([0], [0], label='16384 shots (no knitting)', marker='o', color=hank_blue, linestyle='')
    knit_dot = Line2D([0], [0], label='16384 shots (with knitting)', marker='o', color=hank_orange, linestyle='')
    plt.legend(handles=[no_knit_band, knit_band, exact_line, no_knit_dot, knit_dot], loc='lower right', ncol=2, bbox_to_anchor=(0.95, -0.55))
    
    # Aspect ratio
    ratio = 4/9
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    # Add vertical lines
    ax.autoscale(False)
    ax.plot([0.1, 0.1], [0.45, 0.6], color='black', linewidth=0.75)
    ax.plot([0.9, 0.9], [0.45, 0.6], color='black', linewidth=0.75)
    
    # Save
    plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    print(f"Plot saved to: {OUTPUT_FILE}")
    # Don't show in non-interactive mode
    if os.environ.get('DISPLAY'):
        plt.show()


if __name__ == '__main__':
    main()
