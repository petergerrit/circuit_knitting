#!/usr/bin/env python
"""Run step 1, no noise, knitted simulation 10 times with 1024 shots and random seeds.

Saves fermion number and bootstrap error results to JSON file.
"""

import sys
import os
import json
import numpy as np

# Add parent directory to path so circuit_utils can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_utils.params import trot_step_1, mid
from circuit_utils.utils import circuit_knitter, fermion_number, bootstrap_error

# Configuration
num_shots = 1024
num_runs = 10
results_dir = "results"

os.makedirs(results_dir, exist_ok=True)

# Set seed for reproducibility of random seed generation
np.random.seed(42)

# Results list
results = []

for i in range(num_runs):
    sim_seed = int(np.random.randint(0, 1024**2))
    tp_seed = int(np.random.randint(0, 1024**2))
    bs_seed = int(np.random.randint(0, 1024**2))
    
    res = circuit_knitter(trot_step_1, 0, 10, num_shots,
                          simulator_seed=sim_seed,
                          transpiler_seed=tp_seed,
                          noise=False)
    fn = fermion_number(res, mid)
    be = bootstrap_error(res, mid, num_shots, seed=bs_seed)
    
    summary = {
        "run": i + 1,
        "simulator_seed": sim_seed,
        "transpiler_seed": tp_seed,
        "bootstrap_seed": bs_seed,
        "num_shots": num_shots,
        "fermion_number": fn,
        "bootstrap_error": be,
        "knitted": True,
        "trotter_step": 1
    }
    results.append(summary)
    
    print(f"Run {i+1}/{num_runs}: fermion_number={fn:.6f}, bootstrap_error={be:.6f}")

# Build final output
output = {
    "experiment": "step1_knitted_1024shots_10runs",
    "num_shots": num_shots,
    "num_runs": num_runs,
    "results": results
}

# Save results
results_filename = os.path.join(results_dir, "step1_knitted_10runs.json")
with open(results_filename, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {results_filename}")
