#!/usr/bin/env python3
"""
Run parameter fixing experiments for circuit knitting.

Generates data for mass/epsilon scans matching Hank's notebook.
Saves raw data to data/ directory for plotting.
"""

import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.knitter import circuit_knitter
from config import ExperimentConfig

# Configuration - matching Hank's notebook and production scripts
Nqbits = 12
insertion_point = 4
num_shots = 1024 * 16  # 16384
num_runs = 10

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def run_noisy_experiment(epsilon, mass, trotter_step, num_runs=10):
    """
    Run non-knitted experiment with noise for given parameters.
    Returns list of {fermion_number, bootstrap_error} dictionaries.
    """
    from qiskit import transpile
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
    from qiskit_ibm_runtime import SamplerV2

    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    circuit.measure_all()

    results = []
    for i in range(num_runs):
        simulator_seed = random.randint(0, 2**31 - 1)
        transpiler_seed = random.randint(0, 2**31 - 1)
        bootstrap_seed = random.randint(0, 2**31 - 1)

        backend = FakeWashingtonV2()
        pass_manager = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            seed_transpiler=transpiler_seed
        )
        transpiled_circuit = pass_manager.run(circuit)

        options = {
            "simulator": {
                "seed_simulator": simulator_seed
            }
        }
        sampler = SamplerV2(backend, options=options)

        job = sampler.run([transpiled_circuit], shots=num_shots)
        result = job.result()[0]
        counts = result.data.meas.get_counts()

        fn = fermion_number(counts, insertion_point)
        boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bootstrap_seed)

        results.append({
            "fermion_number": fn,
            "bootstrap_error": boot_err
        })

    return results


def run_knitted_experiment(epsilon, mass, trotter_step, num_runs=10):
    """
    Run knitted experiment for given parameters.
    Returns list of {fermion_number, bootstrap_error} dictionaries.
    """
    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    circuit.measure_all()

    results = []
    for i in range(num_runs):
        simulator_seed = random.randint(0, 2**31 - 1)
        transpiler_seed = random.randint(0, 2**31 - 1)
        bootstrap_seed = random.randint(0, 2**31 - 1)

        config = ExperimentConfig(noise=True)
        knitted_results = circuit_knitter(
            circuit=circuit,
            start_qubit=0,
            end_qubit=10,
            num_shots=num_shots,
            config=config,
            simulator_seed=simulator_seed,
            transpiler_seed=transpiler_seed
        )

        counts = knitted_results['results']
        fn = fermion_number(counts, insertion_point)
        boot_err = bootstrap_error(counts, insertion_point, num_shots, seed=bootstrap_seed)

        results.append({
            "fermion_number": fn,
            "bootstrap_error": boot_err
        })

    return results


def compute_averages(results):
    """Compute mean fermion_number and bootstrap_error from per-run results."""
    fermion_numbers = [r["fermion_number"] for r in results]
    bootstrap_errors = [r["bootstrap_error"] for r in results]
    return {
        "mean_fermion_number": np.mean(fermion_numbers),
        "mean_bootstrap_error": np.mean(bootstrap_errors),
        "fermion_numbers": fermion_numbers,
        "bootstrap_errors": bootstrap_errors
    }


def save_data(filename, data):
    """Save data to JSON file in data directory."""
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)


def save_aggregated_data(filename, data):
    """Save aggregated (averaged) data to JSON file."""
    with open(os.path.join(DATA_DIR, f"agg_{filename}"), 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Run parameter fixing experiments and save data."""
    masses = [1.0, 1.125, 1.25]
    trotter_steps = [0, 1, 2]

    # Aggregated data storage for plotting
    mass_scan_data = {"m1p0": {"fermion_number": [], "bootstrap_error": []},
                       "m1p125": {"fermion_number": [], "bootstrap_error": []},
                       "m1p25": {"fermion_number": [], "bootstrap_error": []}}
    full_steps = [0.2 * i for i in range(33)]

    # Mass scan data at epsilon=0.005
    for mass in masses:
        mass_key = f"m{str(mass).replace('.', 'p')}"
        for trotter_step in trotter_steps:
            results = run_noisy_experiment(
                epsilon=0.005,
                mass=mass,
                trotter_step=trotter_step,
                num_runs=num_runs
            )
            
            # Save raw data
            step_str = f"step{trotter_step}"
            filename = f"data_{step_str}_eps0p005_{mass_key}.json"
            save_data(filename, {
                "epsilon": 0.005,
                "mass": mass,
                "trotter_step": trotter_step,
                "results": results
            })
            
            # Compute and store averages for plotting
            avg = compute_averages(results)
            mass_scan_data[mass_key]["fermion_number"].append(avg["mean_fermion_number"])
            mass_scan_data[mass_key]["bootstrap_error"].append(avg["mean_bootstrap_error"])
    
    # Save aggregated mass scan data
    save_aggregated_data("mass_scan_eps0p005.json", {
        "epsilon": 0.005,
        "steps": full_steps[:len(mass_scan_data["m1p0"]["fermion_number"])],
        "data": mass_scan_data
    })

    # Trotter vs exact comparison for m=1.125
    trotter_vs_exact_data = {
        "exact": {"fermion_number": [], "bootstrap_error": []},
        "trotter": {"fermion_number": [], "bootstrap_error": []}
    }
    short_steps = np.linspace(0, 1.6, 9).tolist()

    for trotter_step in trotter_steps:
        exact_results = run_noisy_experiment(
            epsilon=0.005,
            mass=1.125,
            trotter_step=trotter_step,
            num_runs=num_runs
        )
        knitted_results = run_knitted_experiment(
            epsilon=0.005,
            mass=1.125,
            trotter_step=trotter_step,
            num_runs=num_runs
        )
        
        # Save raw data
        filename = f"trotter_vs_exact_step{trotter_step}_m1p125.json"
        save_data(filename, {
            "epsilon": 0.005,
            "mass": 1.125,
            "trotter_step": trotter_step,
            "exact": exact_results,
            "trotter": knitted_results
        })
        
        # Compute and store averages for plotting
        exact_avg = compute_averages(exact_results)
        trotter_avg = compute_averages(knitted_results)
        trotter_vs_exact_data["exact"]["fermion_number"].append(exact_avg["mean_fermion_number"])
        trotter_vs_exact_data["exact"]["bootstrap_error"].append(exact_avg["mean_bootstrap_error"])
        trotter_vs_exact_data["trotter"]["fermion_number"].append(trotter_avg["mean_fermion_number"])
        trotter_vs_exact_data["trotter"]["bootstrap_error"].append(trotter_avg["mean_bootstrap_error"])
    
    # Save aggregated trotter vs exact data
    save_aggregated_data("trotter_vs_exact_m1p125.json", {
        "mass": 1.125,
        "epsilon": 0.005,
        "steps": short_steps[:len(trotter_vs_exact_data["exact"]["fermion_number"])],
        "data": trotter_vs_exact_data
    })


if __name__ == "__main__":
    main()
