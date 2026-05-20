#!/usr/bin/env python3
"""
Run 10 separate runs of first Trotter step with epsilon=0.2 and 16384 shots.

Each run uses different random seeds for simulator_seed, transpiler_seed, and bootstrap_seed.
Results are saved in JSON format.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.execution import run_circuit_experiment
from config import ExperimentConfig


# Configuration parameters
Nqbits = 12
epsilon = 0.2
mass = 1.125
insertion_point = 4
num_shots = 1024 * 16
num_runs = 10
results_dir = "results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create first Trotter step circuit (step 1)
circuit = trotter_stepper(1, Nqbits, epsilon, mass, insertion_point)


def run_single(simulator_seed, transpiler_seed, bootstrap_seed):
    """Run circuit once and return result summary dictionary."""
    config = ExperimentConfig(noise=False)
    
    counts = run_circuit_experiment(
        circuit=circuit,
        config=config,
        simulator_seed=simulator_seed,
        transpiler_seed=transpiler_seed
    )
    
    fn = fermion_number(counts, insertion_point)
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bootstrap_seed)
    
    return {
        "simulator_seed": simulator_seed,
        "transpiler_seed": transpiler_seed,
        "bootstrap_seed": bootstrap_seed,
        "trotter_step": 1,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "knitted": False,
        "fermion_number": fn,
        "bootstrap_error": boot_err
    }


if __name__ == "__main__":
    # Run with different random seeds for each seed type
    all_results = []
    for i in range(num_runs):
        simulator_seed = random.randint(0, 2**31 - 1)
        transpiler_seed = random.randint(0, 2**31 - 1)
        bootstrap_seed = random.randint(0, 2**31 - 1)
        result = run_single(simulator_seed, transpiler_seed, bootstrap_seed)
        all_results.append(result)
    
    # Build combined output
    combined_output = {
        "experiment": "step1_eps0p2",
        "epsilon": epsilon,
        "num_shots": num_shots,
        "num_runs": num_runs,
        "trotter_step": 1,
        "Nqbits": Nqbits,
        "mass": mass,
        "insertion_point": insertion_point,
        "results": all_results
    }
    
    # Save to JSON file
    output_filename = os.path.join(results_dir, "step1_eps0p2.json")
    with open(output_filename, 'w') as f:
        json.dump(combined_output, f, indent=2)
