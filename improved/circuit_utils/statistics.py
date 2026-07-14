#!/usr/bin/env python3
"""
Statistical analysis module for quantum circuit measurement results.

This module provides functions for computing fermion number and bootstrap errors
from measurement counts.
"""

import numpy as np


def fermion_number(counts, insertion_point):
    """Calculate mean fermion number from measurement counts.

    Args:
        counts (dict): Dictionary of measurement outcomes (bitstrings) to counts
        insertion_point (int): Insertion point index; fermion number is determined from bit at position insertion_point+1

    Returns:
        float: Mean fermion number
    """
    mean = 0
    total_counts = sum(counts.values())
    for s in counts:
        p = s[insertion_point+1]
        if p == '1':
            mean += 1./total_counts * counts[s]
    return mean


def bootstrap_error(counts, insertion_point, shots, seed=1):
    """Calculate bootstrap error for fermion number measurement.

    Args:
        counts (dict): Dictionary of measurement outcomes (bitstrings) to counts
        insertion_point (int): Insertion point index; fermion number is determined from bit at position insertion_point+1
        shots (int): Number of shots used in the original experiment
        seed (int): Random seed for reproducibility (default: 1)

    Returns:
        float: Bootstrap standard deviation (error estimate)
    """
    np.random.seed(seed)
    nshots = shots
    B = 100
    k = list(counts.keys())
    prob = [np.abs(counts[a]) for a in k]
    means = []
    for b in range(B):
        m = 0
        samples = np.random.choice(k, size=nshots, p=(prob / sum(prob)))
        for s in samples:
            p = s[insertion_point+1]
            if p == '1' and counts[s] > 0:
                m += 1./nshots
            elif p == '1':
                m -= 1./nshots
        means.append(m)
    return float(np.std(means))
