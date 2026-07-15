#!/usr/bin/env python3
"""
Run 10 separate runs of second Trotter step with epsilon=0.5 and 16384 shots.

Each run uses different random seeds for simulator_seed, transpiler_seed, and bootstrap_seed.
Results are saved in JSON format matching step2_knitted_all_shots.json structure.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.knitter import circuit_knitter
from config import ExperimentConfig


# Configuration parameters
Nqbits = 12
epsilon = 0.5
mass = 1.125
insertion_point = 4
num_shots = 1024 * 16
num_runs = 10
results_dir = "results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)
circuit.measure_all()


def run_single_knitted(simulator_seed, transpiler_seed, bootstrap_seed):
    """Run circuit knitter once and return result summary dictionary."""
    config = ExperimentConfig(noise=True)
    
    knitted_results = circuit_knitter(
        circuit=circuit,
        start_qubit=0,
        end_qubit=10,
        num_shots=num_shots,
        config=config,
        simulator_seed=simulator_seed,
        transpiler_seed=transpiler_seed
    )
    
    counts = knitted_results['results']
    
    fn = fermion_number(counts, insertion_point)
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bootstrap_seed)
    
    return {
        "simulator_seed": simulator_seed,
        "transpiler_seed": transpiler_seed,
        "bootstrap_seed": bootstrap_seed,
        "trotter_step": 2,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "knitted": True,
        "start_qubit": 0,
        "end_qubit": 10,
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
        result = run_single_knitted(simulator_seed, transpiler_seed, bootstrap_seed)
        all_results.append(result)
    
    # Build combined output matching step2_knitted_all_shots.json structure
    combined_output = {
        "experiment": "step2_knitted_eps0p5",
        "epsilon": epsilon,
        "num_shots": num_shots,
        "num_runs": num_runs,
        "trotter_step": 2,
        "Nqbits": Nqbits,
        "mass": mass,
        "insertion_point": insertion_point,
        "results": all_results
    }
    
    # Save to JSON file
    output_filename = os.path.join(results_dir, "step2_knitted_eps0p5.json")
    with open(output_filename, 'w') as f:
        json.dump(combined_output, f, indent=2)
