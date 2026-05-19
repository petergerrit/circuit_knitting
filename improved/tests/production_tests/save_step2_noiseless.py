#!/usr/bin/env python3
"""
Save second Trotter step (step 2) fermion number measurement and bootstrap error to file.

This script creates a circuit with Trotter step 2 preparation,
measures it, calculates the fermion number and its bootstrap error,
and saves the results to files for later plotting.
"""

import sys
import os
import json

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
num_shots = 1024**2
starting_shots = 1024
powers_of_two = 4
results_dir = "results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)


def save_step2_fermion_number():
    """Create step 2 circuit, measure fermion number, compute bootstrap error, and save to file."""
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
    # print(f"Fermion number: {fn}")
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=42)
    # print(f"Bootstrap error: {boot_err}")
    
    # Save summary results to JSON
    sim_seed = 42
    bs_seed = 42
    summary = {
        "simulator_seed": sim_seed,
        "bootstrap_seed": bs_seed,
        "trotter_step": 2,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "fermion_number": fn,
        "bootstrap_error": boot_err
    }
    summary_filename = os.path.join(results_dir, f"step2_summary_seed{sim_seed}.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    # print(f"Saved summary to {summary_filename}")
    
    return fn, boot_err, summary_filename


def save_step2_fermion_number_knitted(num_shots):
    """Run the step 2 circuit through circuit knitter, measure fermion number, compute bootstrap error, and return results.
    
    Args:
        num_shots: Number of shots to run the circuit with.
    
    Returns:
        dict: Summary containing fermion number and bootstrap error for the given shot count.
    """
    
    # Create config
    config = ExperimentConfig(noise=False)
    
    # Run through circuit knitter (knitting qubits 0 and 10)
    knitted_results = circuit_knitter(
        circuit=circuit,
        start_qubit=0,
        end_qubit=10,
        num_shots=num_shots,
        config=config,
        simulator_seed=42,
        transpiler_seed=42
    )
    
    counts = knitted_results['results']
    
    # Calculate fermion number
    fn = fermion_number(counts, insertion_point)
    # print(f"Knitted fermion number: {fn}")
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=42)
    # print(f"Knitted bootstrap error: {boot_err}")
    
    sim_seed = 42
    tp_seed = 42
    bs_seed = 42
    
    summary = {
        "simulator_seed": sim_seed,
        "transpiler_seed": tp_seed,
        "bootstrap_seed": bs_seed,
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
    
    return summary


if __name__ == "__main__":
    # print("Running second Trotter step fermion number analysis...")
    fermion_num, error, summary_file = save_step2_fermion_number()
    # print(f"\nSecond Trotter step fermion number: {fermion_num:.4f} +/- {error:.4f}")
    
    # Run knitted analysis with varying shots and collect all results
    # print("\nRunning knitted second Trotter step fermion number analysis with varying shots...")
    all_knitted_results = []
    for power in range(powers_of_two + 1):
        current_shots = starting_shots * 2**power
        summary = save_step2_fermion_number_knitted(current_shots)
        all_knitted_results.append(summary)
        # print(f"\nKnitted second Trotter step fermion number ({current_shots} shots): {summary['fermion_number']:.4f} +/- {summary['bootstrap_error']:.4f}")
    
    # Save all knitted results to a single JSON file
    combined_summary = {
        "experiment": "step2_knitted_varying_shots",
        "starting_shots": starting_shots,
        "powers_of_two": powers_of_two,
        "results": all_knitted_results
    }
    combined_filename = os.path.join(results_dir, "step2_knitted_all_shots.json")
    with open(combined_filename, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    # print(f"\nSaved all knitted results to {combined_filename}")
    
    # print("\nAll results saved to the 'results/' directory.")
