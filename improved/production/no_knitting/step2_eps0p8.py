#!/usr/bin/env python3
"""
Run 10 separate runs of second Trotter step with epsilon=0.8 and 16384 shots.

Each run uses different random seeds for simulator_seed, transpiler_seed, and bootstrap_seed.
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
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error


# Configuration parameters
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4
num_shots = 1024 * 16
num_runs = 10
results_dir = "results"
optimization_level = 3

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)
circuit.measure_all()


def run_single(simulator_seed, transpiler_seed, bootstrap_seed):
    """Run circuit once and return result summary dictionary."""
    noise = True
    
    # Set up backend based on noise configuration
    backend = FakeWashingtonV2() if noise else AerSimulator()
    
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
        "trotter_step": 2,
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
        "experiment": "step2_eps0p8",
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
    output_filename = os.path.join(results_dir, "step2_eps0p8.json")
    with open(output_filename, 'w') as f:
        json.dump(combined_output, f, indent=2)
