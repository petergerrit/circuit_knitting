#!/usr/bin/env python3
"""
Script to investigate randomness in circuit knitter.
Runs with simulator_seed=42, transpiler_seed=42 repeatedly until killed.
Appends fermion numbers to a text file after each run.

Usage:
    python investigate_randomness.py --shots 16 --trotter-step 2
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number
from knitter.knitter import circuit_knitter
from config import ExperimentConfig

# Fixed configuration
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4


def run_and_log(iteration, circuit, num_shots, config, output_file):
    """Run knitter and log fermion number."""
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
    fn = fermion_number(counts, insertion_point)

    timestamp = datetime.now().isoformat()
    log_line = f"{timestamp} | iteration={iteration} | fermion_number={fn}\n"

    with open(output_file, 'a') as f:
        f.write(log_line)

    return fn


def main():
    parser = argparse.ArgumentParser(description='Investigate randomness in circuit knitter.')
    parser.add_argument('--shots', type=int, default=16, help='Number of shots (default: 16)')
    parser.add_argument('--trotter-step', type=int, default=2, help='Trotter step (default: 2)')
    args = parser.parse_args()

    # Dynamic configuration from arguments
    num_shots_knitted = args.shots
    trotter_step = args.trotter_step
    OUTPUT_FILE = f"fermi_number_investigation_shots{num_shots_knitted}_step{trotter_step}.txt"

    # Create Trotter step circuit
    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    config = ExperimentConfig(noise=True)

    iteration = 0
    unique_results = set()

    try:
        while True:
            iteration += 1
            fn = run_and_log(iteration, circuit, num_shots_knitted, config, OUTPUT_FILE)
            unique_results.add(fn)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
