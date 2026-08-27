#!/usr/bin/env python3
"""
Run trotter comparison experiment for circuit knitting.

Produces two curves:
- Curve 1: epsilon=0.05, trotter_steps=[0,4,8,...,32], 9 points
- Curve 2: mixed trotter steps and epsilons, 9 points
Total: 18 data points

Each point calculated with 1048576 shots, noiseless simulation.
Bootstrap error is stored for each point.
Data is saved sequentially after each run.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

# Configuration
Nqbits = 12
insertion_point = 4
num_shots = 1024 ** 2

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Output filename
OUTPUT_FILE = os.path.join(DATA_DIR, "trotter_comp.json")


def run_noiseless_simulation(epsilon, mass, trotter_step, num_shots=1024 ** 2, seed=None):
    """
    Run a single noiseless simulation for given parameters.
    Returns (fermion_number, bootstrap_error).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    circuit.measure_all()

    backend = AerSimulator()
    transpiled_circuit = transpile(circuit, backend)
    
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()

    fn = fermion_number(counts, insertion_point)
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=seed)
    
    return fn, boot_err


def save_data(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Run trotter comparison experiment."""
    mass = 1.0  # Use fixed mass for both curves
    
    # Curve 1: epsilon=0.05, trotter_steps=[0,4,8,...,32], 9 points
    # time = trotter_step * 0.05
    curve1 = {
        "epsilon": 0.05,
        "mass": mass,
        "points": []
    }
    trotter_steps_curve1 = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    
    # Curve 2: mixed parameters, 9 points
    curve2 = {
        "mass": mass,
        "points": []
    }
    # 1 point: trotter_step=0, arbitrary epsilon (use 0.1)
    curve2_points = [
        {"trotter_step": 0, "epsilon": 0.1, "time": 0.0},
    ]
    # 4 points: trotter_step=1, epsilon=0.2,0.4,0.6,0.8, times=0.2,0.4,0.6,0.8
    curve2_points += [
        {"trotter_step": 1, "epsilon": 0.2, "time": 0.2},
        {"trotter_step": 1, "epsilon": 0.4, "time": 0.4},
        {"trotter_step": 1, "epsilon": 0.6, "time": 0.6},
        {"trotter_step": 1, "epsilon": 0.8, "time": 0.8},
    ]
    # 4 points: trotter_step=2, epsilon=0.5,0.6,0.7,0.8, times=1,1.2,1.4,1.6
    curve2_points += [
        {"trotter_step": 2, "epsilon": 0.5, "time": 1.0},
        {"trotter_step": 2, "epsilon": 0.6, "time": 1.2},
        {"trotter_step": 2, "epsilon": 0.7, "time": 1.4},
        {"trotter_step": 2, "epsilon": 0.8, "time": 1.6},
    ]
    
    all_data = {
        "num_shots": num_shots,
        "noiseless": True,
        "Nqbits": Nqbits,
        "insertion_point": insertion_point,
        "curve1": curve1,
        "curve2": curve2
    }
    
    # Run Curve 1 (9 points)
    for i, trotter_step in enumerate(trotter_steps_curve1):
        time = trotter_step * 0.05
        seed = random.randint(0, 2**31 - 1)
        
        fn, boot_err = run_noiseless_simulation(
            epsilon=0.05,
            mass=mass,
            trotter_step=trotter_step,
            num_shots=num_shots,
            seed=seed
        )
        
        curve1["points"].append({
            "trotter_step": trotter_step,
            "time": time,
            "fermion_number": fn,
            "bootstrap_error": boot_err,
            "seed": seed
        })
        
        # Save after each point
        save_data(all_data, OUTPUT_FILE)
    
    # Run Curve 2 (9 points)
    for i, params in enumerate(curve2_points):
        seed = random.randint(0, 2**31 - 1)
        
        fn, boot_err = run_noiseless_simulation(
            epsilon=params["epsilon"],
            mass=mass,
            trotter_step=params["trotter_step"],
            num_shots=num_shots,
            seed=seed
        )
        
        curve2["points"].append({
            "trotter_step": params["trotter_step"],
            "epsilon": params["epsilon"],
            "time": params["time"],
            "fermion_number": fn,
            "bootstrap_error": boot_err,
            "seed": seed
        })
        
        # Save after each point
        save_data(all_data, OUTPUT_FILE)
    


if __name__ == "__main__":
    main()
