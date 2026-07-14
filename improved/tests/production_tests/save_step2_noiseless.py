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
powers_of_two = 10  # 1024 * 2^10 = 1024^2, so runs: 1024, 2048, ..., 1048576 (11 runs)
results_dir = "results"
base_seed = 42

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)


def save_step2_fermion_number(sim_seed, bs_seed):
    """Create step 2 circuit, measure fermion number, compute bootstrap error, and save to file.
    
    Args:
        sim_seed: Seed for the simulator
        bs_seed: Seed for bootstrap error calculation
        
    Returns:
        tuple: (fermion_number, bootstrap_error, summary)
    """
    circuit.measure_all()
    
    # Run on simulator
    backend = AerSimulator()
    backend.set_options(seed_simulator=sim_seed)
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
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bs_seed)
    # print(f"Bootstrap error: {boot_err}")
    
    # Build summary
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
        "bootstrap_error": boot_err,
        "knitted": False
    }
    
    return fn, boot_err, summary


def save_step2_fermion_number_knitted(num_shots, sim_seed, tp_seed, bs_seed):
    """Run the step 2 circuit through circuit knitter, measure fermion number, compute bootstrap error.
    
    Args:
        num_shots: Number of shots to run the circuit with.
        sim_seed: Seed for the simulator.
        tp_seed: Seed for the transpiler.
        bs_seed: Seed for bootstrap error calculation.
    
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
        simulator_seed=sim_seed,
        transpiler_seed=tp_seed
    )
    
    counts = knitted_results['results']
    
    # Calculate fermion number
    fn = fermion_number(counts, insertion_point)
    # print(f"Knitted fermion number: {fn}")
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bs_seed)
    # print(f"Knitted bootstrap error: {boot_err}")
    
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
    base_sim_seed = base_seed
    base_bs_seed = base_seed + 1
    fermion_num, error, nonknitted_summary = save_step2_fermion_number(base_sim_seed, base_bs_seed)
    # print(f"\nSecond Trotter step fermion number: {fermion_num:.4f} +/- {error:.4f}")
    
    # Save non-knitted result to its own file
    nonknitted_filename = os.path.join(results_dir, "step2_nonknitted.json")
    with open(nonknitted_filename, 'w') as f:
        json.dump(nonknitted_summary, f, indent=2)
    
    # Run knitted analysis with varying shots and save incrementally
    # print("\nRunning knitted second Trotter step fermion number analysis with varying shots...")
    combined_summary = {
        "experiment": "step2_knitted_varying_shots",
        "base_seed": base_seed,
        "starting_shots": starting_shots,
        "powers_of_two": powers_of_two,
        "results": []
    }
    combined_filename = os.path.join(results_dir, "step2_knitted.json")
    
    for power in range(powers_of_two + 1):
        current_shots = starting_shots * 2**power
        # Generate unique seeds for this run
        sim_seed = base_seed + 100 + power * 3
        tp_seed = base_seed + 101 + power * 3
        bs_seed = base_seed + 102 + power * 3
        
        summary = save_step2_fermion_number_knitted(current_shots, sim_seed, tp_seed, bs_seed)
        combined_summary["results"].append(summary)
        # print(f"\nKnitted second Trotter step fermion number ({current_shots} shots): {summary['fermion_number']:.4f} +/- {summary['bootstrap_error']:.4f}")
        
        # Save results so far (incremental update)
        with open(combined_filename, 'w') as f:
            json.dump(combined_summary, f, indent=2)
    
    # print(f"\nSaved all knitted results to {combined_filename}")
    
    # print("\nAll results saved to the 'results/' directory.")
