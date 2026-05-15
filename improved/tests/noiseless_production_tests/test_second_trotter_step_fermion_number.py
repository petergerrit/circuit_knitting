#!/usr/bin/env python3
"""
Test for second Trotter step (step 2) fermion number measurement and bootstrap error.

This test creates a circuit with Trotter step 2 preparation,
measures it, and calculates the fermion number and its bootstrap error.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.knitter import circuit_knitter
from config import ExperimentConfig


# Configuration parameters (from parent directory params.ipynb)
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4  # Insertion point index for meson operator
num_shots = 1024#**2
num_shots_knitted = 1024

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)


def test_second_trotter_step_fermion_number():
    """Create step 2 circuit, measure fermion number, and compute bootstrap error."""
    circuit.measure_all()
    
    # Run on simulator
    backend = AerSimulator()
    backend.set_options(seed_simulator=42)
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled_circuit = pass_manager.run(circuit)
    
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    
    # Calculate fermion number
    fn = fermion_number(counts, insertion_point)
    print(f"Fermion number: {fn}")
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=42)
    print(f"Bootstrap error: {boot_err}")
    
    return fn, boot_err


def test_second_trotter_step_fermion_number_knitted():
    """Run the step 2 circuit through circuit knitter, measure fermion number, and compute bootstrap error."""
    
    # Create config
    config = ExperimentConfig(noise=False)
    
    # Run through circuit knitter (knitting qubits 0 and 10)
    knitted_results = circuit_knitter(
        circuit=circuit,
        start_qubit=0,
        end_qubit=10,
        num_shots=num_shots_knitted,
        config=config,
        simulator_seed=42,
        transpiler_seed=42
    )
    
    counts = knitted_results['results']
    
    # Calculate fermion number
    fn = fermion_number(counts, insertion_point)
    print(f"Knitted fermion number: {fn}")
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots_knitted, seed=42)
    print(f"Knitted bootstrap error: {boot_err}")
    
    return fn, boot_err


if __name__ == "__main__":
    fermion_num, error = test_second_trotter_step_fermion_number()
    print(f"\nSecond Trotter step fermion number: {fermion_num:.4f} +/- {error:.4f}")
    
    knitted_fn, knitted_err = test_second_trotter_step_fermion_number_knitted()
    print(f"\nKnitted second Trotter step fermion number: {knitted_fn:.4f} +/- {knitted_err:.4f}")
