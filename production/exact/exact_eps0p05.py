#!/usr/bin/env python3
"""
Run exact simulations for all Trotter steps from 0 to 32 with epsilon=0.05 and 1,048,576 shots.

This produces exact (noise-free) results at each time step t = 0, 0.05, 0.10, ..., 1.6
for comparison with the knitted and no_knitting approximations.

Each step uses a single run with 1,048,576 shots (1024**2) for high-precision results.
Results are saved in JSON format.
"""

import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error


# Configuration parameters
Nqbits = 12
epsilon = 0.05
mass = 1.125
insertion_point = 4
num_shots = 1048576  # 1024**2 = 1,048,576 shots per step
num_runs = 1
results_dir = "results"
optimization_level = 3

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# All Trotter steps: 0 to 32 (33 steps total, t=0 to t=1.6 in steps of 0.05)
all_steps = list(range(33))


def run_single(step, simulator_seed, transpiler_seed, bootstrap_seed):
    """Run a single Trotter step circuit once and return result summary dictionary."""
    noise = False  # Exact simulation - no noise
    
    # Set up backend (AerSimulator for noise-free)
    backend = AerSimulator()
    
    # Create circuit for this specific Trotter step
    circuit = trotter_stepper(step, Nqbits, epsilon, mass, insertion_point)
    circuit.measure_all()
    
    # Set up transpiler
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed or np.random.randint(1024**2)
    )
    
    # Transpile circuit
    transpiled_circuit = pass_manager.run(circuit)
    
    # Set up sampler with options
    options = {
        "simulator": {
            "seed_simulator": simulator_seed or np.random.randint(1024**2)
        }
    }
    sampler = SamplerV2(backend, options=options)
    
    # Run job and get results
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    
    fn = fermion_number(counts, insertion_point)
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bootstrap_seed)
    
    return {
        "simulator_seed": simulator_seed,
        "transpiler_seed": transpiler_seed,
        "bootstrap_seed": bootstrap_seed,
        "trotter_step": step,
        "time": step * epsilon,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "knitted": False,
        "noise": False,
        "fermion_number": fn,
        "bootstrap_error": boot_err
    }


def run_step(step):
    """Run a single Trotter step with high shot count and return results."""
    simulator_seed = random.randint(0, 2**31 - 1)
    transpiler_seed = random.randint(0, 2**31 - 1)
    bootstrap_seed = random.randint(0, 2**31 - 1)
    result = run_single(step, simulator_seed, transpiler_seed, bootstrap_seed)
    
    return {
        "trotter_step": step,
        "time": step * epsilon,
        "epsilon": epsilon,
        "num_shots": num_shots,
        "num_runs": num_runs,
        "Nqbits": Nqbits,
        "mass": mass,
        "insertion_point": insertion_point,
        "result": result
    }


if __name__ == "__main__":
    # Run all 33 Trotter steps
    all_step_results = []
    
    for step in all_steps:
        print(f"Running step {step} (t={step * epsilon:.2f})...")
        step_results = run_step(step)
        all_step_results.append(step_results)
        print(f"  Completed step {step}")
    
    # Build combined output
    combined_output = {
        "experiment": "exact_eps0p05_full_evolution",
        "epsilon": epsilon,
        "num_shots": num_shots,
        "num_runs": num_runs,
        "Nqbits": Nqbits,
        "mass": mass,
        "insertion_point": insertion_point,
        "steps": all_step_results
    }
    
    # Save to JSON file
    output_filename = os.path.join(results_dir, "exact_eps0p05.json")
    with open(output_filename, 'w') as f:
        json.dump(combined_output, f, indent=2)
    
    print(f"\nAll steps complete. Results saved to {output_filename}")
