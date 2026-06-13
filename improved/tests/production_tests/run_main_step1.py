#!/usr/bin/env python3
"""
Run step 1 from main.py (no noise, knitted) with epsilon=0.8
Outputs fermion numbers to run_main_step1_results.txt as they compute
"""

import sys
import os
import atexit

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from circuits.basic_circuits import trotter_stepper
from knitter.knitter import circuit_knitter
from circuit_utils.statistics import fermion_number
from config import ExperimentConfig

# Configuration matching main.py
Nqbits = 12
epsilon = 0.8
mass = 1.125
mid = 4

# Output file
output_file = os.path.join(os.path.dirname(__file__), 'run_main_step1_results.txt')

# Cleanup
def cleanup():
    pass  # Results file is intentionally kept
atexit.register(cleanup)

# Create circuit
trot_step_1 = trotter_stepper(1, Nqbits, epsilon, mass, mid)

# Clear output file
with open(output_file, 'w') as f:
    f.write('# Step 1, no noise, knitted, epsilon=0.8\n')
    f.write('# Shot count: fermion number\n')

np.random.seed(1)
for i in range(11):
    num_shots = 1024 * 2**i
    config = ExperimentConfig(noise=False)
    result = circuit_knitter(
        trot_step_1, 0, 10, num_shots, 
        config=config,
        simulator_seed=np.random.randint(1024**2),
        transpiler_seed=np.random.randint(1024**2)
    )
    counts = result['results']
    fn = fermion_number(counts, mid)
    line = f"{num_shots}: {fn}\n"
    with open(output_file, 'a') as f:
        f.write(line)
    print(line, end='')  # Also print to terminal
    sys.stdout.flush()

print(f"\nResults saved to: {output_file}")
