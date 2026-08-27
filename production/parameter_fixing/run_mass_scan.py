#!/usr/bin/env python3
"""
Run noiseless mass scan for circuit knitting parameter fixing.

Runs simulations for:
- 3 mass values: 1.0, 1.125, 1.25
- epsilon = 0.05 (fixed)
- 33 trotter steps: 0, 4, 8, ..., 128 (corresponding to t = 0, 0.2, ..., 6.4)
- 1024 shots per data point
- Noiseless (AerSimulator without hardware noise model)

Total: 3 masses × 33 steps = 99 data points
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

# Configuration
Nqbits = 12
insertion_point = 4
num_shots = 1024

# Mass values
masses = [1.0, 1.125, 1.25]

# 33 trotter steps: 0, 4, 8, ..., 128 (t = 0, 0.2, 0.4, ..., 6.4)
trotter_steps = [4 * i for i in range(33)]  # 0, 4, 8, ..., 128
time_steps = [0.2 * i for i in range(33)]    # 0.0, 0.2, 0.4, ..., 6.4

epsilon = 0.05

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def run_noiseless_simulation(epsilon, mass, trotter_step, num_shots=1024):
    """
    Run a single noiseless simulation for given parameters.
    Returns fermion_number.
    """
    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    circuit.measure_all()

    # Use AerSimulator directly (noiseless)
    backend = AerSimulator()
    
    # Transpile for the simulator
    transpiled_circuit = transpile(circuit, backend)
    
    # Run with sampler
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()

    fn = fermion_number(counts, insertion_point)
    return fn


def main():
    """Run mass scan and save data."""
    all_data = {}
    
    for mass in masses:
        mass_key = f"m{str(mass).replace('.', 'p')}"
        all_data[mass_key] = {
            "fermion_numbers": [],
            "trotter_steps": [],
            "time_steps": []
        }
        
        for i, (trotter_step, t) in enumerate(zip(trotter_steps, time_steps)):
            print(f"Running mass={mass}, trotter_step={trotter_step}, t={t}")
            
            fn = run_noiseless_simulation(
                epsilon=epsilon,
                mass=mass,
                trotter_step=trotter_step,
                num_shots=num_shots
            )
            
            all_data[mass_key]["fermion_numbers"].append(fn)
            all_data[mass_key]["trotter_steps"].append(trotter_step)
            all_data[mass_key]["time_steps"].append(t)
    
    # Save raw data
    filename = f"mass_scan_noiseless_eps{epsilon}_shots{num_shots}.json"
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump({
            "epsilon": epsilon,
            "num_shots": num_shots,
            "masses": masses,
            "trotter_steps": trotter_steps,
            "time_steps": time_steps,
            "data": all_data
        }, f, indent=2)
    
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()
