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

# Create first Trotter step circuit (step 1)
circuit = trotter_stepper(1, Nqbits, epsilon, mass, insertion_point)


def save_step1_fermion_number(sim_seed, bs_seed):
    """Create step 1 circuit, measure fermion number, compute bootstrap error, and save to file.
    
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
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bs_seed)
    
    # Build summary
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
        "bootstrap_error": boot_err,
        "knitted": False
    }
    
    return fn, boot_err, summary


def save_step1_fermion_number_knitted(num_shots, sim_seed, tp_seed, bs_seed):
    """Run the step 1 circuit through circuit knitter, measure fermion number, compute bootstrap error.
    
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
    
    # Calculate bootstrap error
    boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bs_seed)
    
    summary = {
        "simulator_seed": sim_seed,
        "transpiler_seed": tp_seed,
        "bootstrap_seed": bs_seed,
        "trotter_step": 1,
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
    base_sim_seed = base_seed
    base_bs_seed = base_seed + 1
    fermion_num, error, nonknitted_summary = save_step1_fermion_number(base_sim_seed, base_bs_seed)
    
    # Save non-knitted result to its own file
    nonknitted_filename = os.path.join(results_dir, "step1_nonknitted.json")
    with open(nonknitted_filename, 'w') as f:
        json.dump(nonknitted_summary, f, indent=2)
    
    # Run knitted analysis with varying shots
    knitted_filename = os.path.join(results_dir, "step1_knitted.json")
    
    # Initialize metadata
    knitted_metadata = {
        "experiment": "step1_knitted_varying_shots",
        "base_seed": base_seed,
        "starting_shots": starting_shots,
        "powers_of_two": powers_of_two,
        "results": []
    }
    
    for power in range(powers_of_two + 1):
        current_shots = starting_shots * 2**power
        # Generate unique seeds for this run
        sim_seed = base_seed + 100 + power * 3
        tp_seed = base_seed + 101 + power * 3
        bs_seed = base_seed + 102 + power * 3
        
        summary = save_step1_fermion_number_knitted(current_shots, sim_seed, tp_seed, bs_seed)
        knitted_metadata["results"].append(summary)
        
        # Save knitted results so far (appending by overwriting with full data)
        with open(knitted_filename, 'w') as f:
            json.dump(knitted_metadata, f, indent=2)
