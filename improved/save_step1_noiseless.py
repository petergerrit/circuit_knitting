#!/usr/bin/env python3
"""
Run first Trotter step (step 1) with knitter, 1024 shots, 10 times with random seeds.

Saves fermion number and bootstrap error results to file.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.knitter import circuit_knitter
from config import ExperimentConfig


# Configuration parameters
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4  # Insertion point index for meson operator
num_shots = 1024
num_runs = 10
results_dir = "results"

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
    # Set random seed for reproducibility of the random seeds themselves
    np.random.seed(42)
    
    # Results file
    results_filename = os.path.join(results_dir, "step1_knitted_10runs.json")
    
    # Initialize results list
    results = []
    
    # Run 10 times with random seeds each time
    for i in range(num_runs):
        sim_seed = int(np.random.randint(0, 2**31))
        tp_seed = int(np.random.randint(0, 2**31))
        bs_seed = int(np.random.randint(0, 2**31))
        
        summary = save_step1_fermion_number_knitted(num_shots, sim_seed, tp_seed, bs_seed)
        results.append(summary)
        
        print(f"Run {i+1}/{num_runs}: fermion_number={summary['fermion_number']:.6f}, "
              f"bootstrap_error={summary['bootstrap_error']:.6f}")
    
    # Build final output with metadata
    output = {
        "experiment": "step1_knitted_1024shots_10runs",
        "Nqbits": Nqbits,
        "epsilon": epsilon,
        "mass": mass,
        "insertion_point": insertion_point,
        "num_shots": num_shots,
        "num_runs": num_runs,
        "results": results
    }
    
    # Save all results
    with open(results_filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_filename}")
