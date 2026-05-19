#!/usr/bin/env python3
"""
Save first Trotter step (step 1) fermion number measurement and bootstrap error to file.

This script creates a circuit with Trotter step 1 preparation,
measures it, calculates the fermion number and its bootstrap error,
and saves the results to files for later plotting.
"""

import sys
import os
import json
import csv

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
num_shots_knitted = 1024*2**4
results_dir = "results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create first Trotter step circuit (step 1)
circuit = trotter_stepper(1, Nqbits, epsilon, mass, insertion_point)


def save_first_trotter_step_fermion_number():
    """Create step 1 circuit, measure fermion number, compute bootstrap error, and save to file."""
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
    
    # Save raw counts to JSON (seed-based naming for reproducibility)
    sim_seed = 42
    bs_seed = 42
    counts_filename = os.path.join(results_dir, f"first_trotter_counts_seed{sim_seed}.json")
    with open(counts_filename, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"Saved counts to {counts_filename}")
    
    # Save summary results to JSON
    summary = {
        "simulator_seed": sim_seed,
        "bootstrap_seed": bs_seed,
        "trotter_step": 1,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "fermion_number": fn,
        "bootstrap_error": boot_err
    }
    summary_filename = os.path.join(results_dir, f"first_trotter_summary_seed{sim_seed}.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_filename}")
    
    # Save counts as CSV for easy plotting
    csv_filename = os.path.join(results_dir, f"first_trotter_counts_seed{sim_seed}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bitstring', 'count'])
        for bitstring, count in counts.items():
            writer.writerow([bitstring, count])
    print(f"Saved counts CSV to {csv_filename}")
    
    # Save each shot on its own line (num_shots total lines)
    shots_filename = os.path.join(results_dir, f"first_trotter_shots_seed{sim_seed}.txt")
    with open(shots_filename, 'w') as f:
        for bitstring, count in counts.items():
            for _ in range(count):
                f.write(bitstring + '\n')
    print(f"Saved {num_shots} shots to {shots_filename}")
    
    return fn, boot_err, counts_filename, summary_filename, csv_filename, shots_filename


def save_first_trotter_step_fermion_number_knitted():
    """Run the step 1 circuit through circuit knitter, measure fermion number, compute bootstrap error, and save."""
    
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
    
    # Save knitted results (seed-based naming for reproducibility)
    sim_seed = 42
    tp_seed = 42
    bs_seed = 42
    
    # Save raw counts to JSON
    counts_filename = os.path.join(results_dir, f"first_trotter_knitted_counts_sim{sim_seed}_tp{tp_seed}_bs{bs_seed}.json")
    with open(counts_filename, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"Saved knitted counts to {counts_filename}")
    
    # Save summary results to JSON
    summary = {
        "simulator_seed": sim_seed,
        "transpiler_seed": tp_seed,
        "bootstrap_seed": bs_seed,
        "trotter_step": 1,
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots_knitted,
        "knitted": True,
        "start_qubit": 0,
        "end_qubit": 10,
        "fermion_number": fn,
        "bootstrap_error": boot_err
    }
    summary_filename = os.path.join(results_dir, f"first_trotter_knitted_summary_sim{sim_seed}_tp{tp_seed}_bs{bs_seed}.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved knitted summary to {summary_filename}")
    
    # Save counts as CSV for easy plotting
    csv_filename = os.path.join(results_dir, f"first_trotter_knitted_counts_sim{sim_seed}_tp{tp_seed}_bs{bs_seed}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bitstring', 'count'])
        for bitstring, count in counts.items():
            writer.writerow([bitstring, count])
    print(f"Saved knitted counts CSV to {csv_filename}")
    
    # Save each shot on its own line (num_shots total lines)
    shots_filename = os.path.join(results_dir, f"first_trotter_knitted_shots_sim{sim_seed}_tp{tp_seed}_bs{bs_seed}.txt")
    with open(shots_filename, 'w') as f:
        for bitstring, count in counts.items():
            for _ in range(int(count)):
                f.write(bitstring + '\n')
    print(f"Saved {num_shots_knitted} knitted shots to {shots_filename}")
    
    return fn, boot_err, counts_filename, summary_filename, csv_filename, shots_filename


if __name__ == "__main__":
    print("Running first Trotter step fermion number analysis...")
    fermion_num, error, counts_file, summary_file, csv_file, shots_file = save_first_trotter_step_fermion_number()
    print(f"\nFirst Trotter step fermion number: {fermion_num:.4f} +/- {error:.4f}")
    
    print("\nRunning knitted first Trotter step fermion number analysis...")
    knitted_fn, knitted_err, k_counts_file, k_summary_file, k_csv_file, k_shots_file = save_first_trotter_step_fermion_number_knitted()
    print(f"\nKnitted first Trotter step fermion number: {knitted_fn:.4f} +/- {knitted_err:.4f}")
    
    print("\nAll results saved to the 'results/' directory.")
