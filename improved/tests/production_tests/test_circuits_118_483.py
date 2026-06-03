#!/usr/bin/env python3
"""
Script to reproduce and test circuits 118 and 483 from step2 debug files.

This script reconstructs the circuit knitting for step2, extracts only circuits
118 and 483, runs them with their original seeds, and outputs results to a new
debug file for comparison with the original debug files.

Expected behavior: Results should match test1 (which used seeds 160 and 525)
"""

import sys
import os

# Add the parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from knitter.knitter import circuit_knitter, fully_decompose_circuit
from knitter.execution import my_measure, comb_measure, flattener, knit_lister
from config import ExperimentConfig
from itertools import product
import numpy as np
from datetime import datetime

# Configuration parameters matching the debug files
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4
num_shots = 16

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)

# Control and target qubits (from debug file header)
conq = 0
tarq = 10

# Fully decompose the circuit
circuit = fully_decompose_circuit(circuit, preserve_unitary=True)

# Count actual CNOTs between control and target
cx_list = [i for i, gate in enumerate(circuit.data) if gate.operation.name == 'cx' and \
           ((circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (tarq, conq) or \
           (circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (conq, tarq))]
num_cx_actual = len(cx_list)

print(f"Found {num_cx_actual} CNOT gates between qubits {conq} and {tarq}")

# Build nest_list by processing the decomposed circuit
nest_list = []
current_cx = 0

for i, gate in enumerate(circuit.data):
    if gate.operation.name == 'cx':
        if (gate.qubits[0]._index, gate.qubits[1]._index) == (tarq, conq):
            nest_list.append(knit_lister(circuit, conq, tarq, current_cx, num_cx_actual))
            current_cx += 1
        elif (gate.qubits[0]._index, gate.qubits[1]._index) == (conq, tarq):
            nest_list.append(knit_lister(circuit, tarq, conq, current_cx, num_cx_actual))
            current_cx += 1
        else:
            nest_list.append([gate])
    elif gate.operation.name not in ("measure", "barrier"):
        nest_list.append([gate])

print(f"nest_list length: {len(nest_list)}")

# Generate all circuits from product
print("Generating all circuits from product...")
circuits_raw = [*product(*nest_list)]
circuits = [flattener(item) for item in circuits_raw]

print(f"Total circuits generated: {len(circuits)}")

# Define the circuits we want to test (from debug files)
target_circuits = {
    118: {'simulator_seed': 160, 'transpiler_seed': 160, 'prefactor': -0.0625},
    483: {'simulator_seed': 525, 'transpiler_seed': 525, 'prefactor': 0.0625},
}

# Configuration
config = ExperimentConfig(noise=True)

# Open debug output file
debug_filename = "debug_circuits_118_483_reproduction.txt"

# Write header
with open(debug_filename, 'w') as debug_fh:
    debug_fh.write(f"# Circuit Knitter Debug Output\n")
    debug_fh.write(f"# Reproduction of circuits 118 and 483 from step2 debug files\n")
    debug_fh.write(f"# Circuit: {circuit.name if circuit.name else 'circuit-46'}\n")
    debug_fh.write(f"# Control qubit: {conq}, Target qubit: {tarq}\n")
    debug_fh.write(f"# Number of CNOTs: {num_cx_actual}\n")
    debug_fh.write(f"# Total circuits to process: {len(circuits)}\n")
    debug_fh.write(f"# Shots per circuit: {num_shots}\n")
    debug_fh.write(f"# Noise: True\n")
    debug_fh.write(f"# Timestamp: {datetime.now().isoformat()}\n")
    debug_fh.write(f"\n# nest_list length: {len(nest_list)}\n")

# Run the target circuits
print("\nRunning target circuits...")
for circuit_idx, circuit_info in target_circuits.items():
    if circuit_idx >= len(circuits):
        print(f"Warning: Circuit {circuit_idx} not found (only {len(circuits)} circuits)")
        continue
    
    seed = circuit_info['simulator_seed']
    transpiler_seed = circuit_info['transpiler_seed']
    prefactor = circuit_info['prefactor']
    
    print(f"\nRunning Circuit {circuit_idx} / {len(circuits)} (prefactor: {prefactor}, seed: {seed})...")
    
    circuit_data = circuits[circuit_idx]
    
    # Execute the circuit
    raw_results = my_measure(
        circuit_data=circuit_data,
        conq=conq,
        tarq=tarq,
        num_qubits=Nqbits,
        num_cx=num_cx_actual,
        num_shots=num_shots,
        simulator_seed=seed,
        transpiler_seed=transpiler_seed,
        noise=True
    )
    
    # Get combined results
    combined_results = comb_measure(raw_results, conq, tarq, num_cx_actual)
    
    # Write to debug file
    with open(debug_filename, 'a') as debug_fh:
        debug_fh.write(f"\nCircuit {circuit_idx} / {len(circuits)} (prefactor: {prefactor})\n")
        debug_fh.write(f"  Simulator seed: {seed}, Transpiler seed: {transpiler_seed}\n")
        debug_fh.write(f"  Raw measurement results (internal): {raw_results}\n")
        debug_fh.write(f"  Combined measurement results: {combined_results}\n")
    
    print(f"  Raw results: {raw_results}")
    print(f"  Combined results: {combined_results}")

print(f"\nDone! Results written to {debug_filename}")
