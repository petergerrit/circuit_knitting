#!/usr/bin/env python3
"""
Compute resampled fermion number arrays for knitting and no_knitting results.

For each of the 9 temporal points (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6),
this script:
1. Loads all JSON results from knitting/results/ and no_knitting/results/
2. Maps each file to its time point: time = trotter_step * epsilon
3. For each time point, collects fermion_number and bootstrap_error from all runs
4. Performs resampling: for each run, generate 1000 samples from N(fermion_number, bootstrap_error)
5. Computes mean and std of pooled samples across all runs
6. Saves resampled arrays as .npy files

This follows the approach used in the final branch's plot_data.ipynb.

Random seeds are fixed for reproducibility.
"""

import os
import json
import glob
import numpy as np

# Fixed random seed for reproducibility
# Single seed for the entire resampling process
RESAMPLING_SEED = 892741056

# Configuration
PRODUCTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNITTING_RESULTS_DIR = os.path.join(PRODUCTION_DIR, "knitting", "results")
NO_KNITTING_RESULTS_DIR = os.path.join(PRODUCTION_DIR, "no_knitting", "results")
OUTPUT_DIR = os.path.join(PRODUCTION_DIR, "resampling")

# 9 target temporal points (same as final branch)
TARGET_TIMES = np.linspace(0, 1.6, 9).tolist()  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

# Tolerance for matching time points (to handle floating point)
TIME_TOLERANCE = 1e-10


def load_json_files(directory):
    """Load all JSON files from a directory and return list of dicts."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data = []
    for f in json_files:
        with open(f, 'r') as file:
            data.append(json.load(file))
    return data


def get_time_from_file(file_data):
    """Compute time = trotter_step * epsilon from file metadata."""
    return file_data['trotter_step'] * file_data['epsilon']


def find_matching_time_index(target_times, time_value):
    """Find index of closest matching time in target_times list."""
    for i, t in enumerate(target_times):
        if abs(time_value - t) < TIME_TOLERANCE:
            return i
    return None


def compute_resampled_arrays(all_results, target_times, num_samples_per_run=1000):
    """
    Compute resampled fermion number and std arrays.
    
    Args:
        all_results: List of dicts, each containing 'results' list with fermion_number and bootstrap_error
        target_times: List of target time points
        num_samples_per_run: Number of samples to draw per run (default 1000)
    
    Returns:
        Tuple of (resampled_fermion_number, resampled_std) arrays
        Note: resampled_std is the standard deviation of the pooled normal samples,
        not a bootstrap error.
    """
    # Group results by time point
    time_to_results = {t: [] for t in target_times}
    
    for file_data in all_results:
        time = get_time_from_file(file_data)
        idx = find_matching_time_index(target_times, time)
        if idx is None:
            print(f"Warning: No matching target time for {time}, skipping")
            continue
        
        target_time = target_times[idx]
        for run in file_data['results']:
            time_to_results[target_time].append({
                'fermion_number': run['fermion_number'],
                'bootstrap_error': run['bootstrap_error']
            })
    
    # Compute resampled arrays
    resampled_fermion_number = []
    resampled_std = []
    
    for t in target_times:
        runs = time_to_results[t]
        if not runs:
            print(f"Warning: No runs found for time {t}, using NaN")
            resampled_fermion_number.append(np.nan)
            resampled_std.append(np.nan)
            continue
        
        # Collect all samples
        all_samples = np.empty(0)
        for run in runs:
            ferm_num = run['fermion_number']
            bs_err = run['bootstrap_error']
            samples = np.random.normal(ferm_num, bs_err, num_samples_per_run)
            all_samples = np.append(all_samples, samples)
        
        # Compute statistics
        resampled_fermion_number.append(np.mean(all_samples))
        resampled_std.append(np.std(all_samples))
        
        print(f"Time {t}: {len(runs)} runs, {len(all_samples)} total samples, "
              f"mean={np.mean(all_samples):.6f}, std={np.std(all_samples):.6f}")
    
    return np.array(resampled_fermion_number), np.array(resampled_std)


def main():
    # Set random seed for reproducibility
    np.random.seed(RESAMPLING_SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data
    print("Loading knitting results...")
    knitting_results = load_json_files(KNITTING_RESULTS_DIR)
    print(f"  Found {len(knitting_results)} files")
    
    print("Loading no_knitting results...")
    no_knitting_results = load_json_files(NO_KNITTING_RESULTS_DIR)
    print(f"  Found {len(no_knitting_results)} files")
    
    # Compute resampled arrays
    print("\nComputing resampled arrays for knitting...")
    knitted_fermion, knitted_std = compute_resampled_arrays(
        knitting_results, TARGET_TIMES
    )
    
    print("\nComputing resampled arrays for no_knitting...")
    no_knit_fermion, no_knit_std = compute_resampled_arrays(
        no_knitting_results, TARGET_TIMES
    )
    
    # Save results
    print("\nSaving results...")
    np.save(os.path.join(OUTPUT_DIR, "knitted_resampled_fermion_number.npy"), knitted_fermion)
    np.save(os.path.join(OUTPUT_DIR, "knitted_resampled_std.npy"), knitted_std)
    np.save(os.path.join(OUTPUT_DIR, "no_knit_resampled_fermion_number.npy"), no_knit_fermion)
    np.save(os.path.join(OUTPUT_DIR, "no_knit_resampled_std.npy"), no_knit_std)
    np.save(os.path.join(OUTPUT_DIR, "trotter_steps.npy"), np.array(TARGET_TIMES))
    
    print(f"\nDone. Saved resampled arrays to {OUTPUT_DIR}/")
    print(f"Target times: {TARGET_TIMES}")


if __name__ == "__main__":
    main()
