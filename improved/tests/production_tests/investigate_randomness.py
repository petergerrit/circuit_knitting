#!/usr/bin/env python3
"""
Script to investigate randomness in circuit knitter.
Runs with simulator_seed=42, transpiler_seed=42 once.
Writes fermion number to the header of the debug output file.

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


def run_and_log(circuit, num_shots, trotter_step, config, debug_file):
    """Run knitter and log fermion number to debug file header."""
    knitted_results = circuit_knitter(
        circuit=circuit,
        start_qubit=0,
        end_qubit=10,
        num_shots=num_shots,
        config=config,
        simulator_seed=42,
        transpiler_seed=42,
        debug_file=debug_file
    )

    counts = knitted_results['results']
    fn = fermion_number(counts, insertion_point)

    timestamp = datetime.now().isoformat()
    header_line = f"# {timestamp} | fermion_number={fn}\n"

    with open(debug_file, 'r+') as f:
        content = f.read()
        f.seek(0)
        f.write(header_line + content)

    return fn


def main():
    parser = argparse.ArgumentParser(description='Investigate randomness in circuit knitter.')
    parser.add_argument('--shots', type=int, default=16, help='Number of shots (default: 16)')
    parser.add_argument('--trotter-step', type=int, default=2, help='Trotter step (default: 2)')
    parser.add_argument('--test-name', type=str, default='baseline', help='Test identifier for output file (default: baseline)')
    args = parser.parse_args()

    # Dynamic configuration from arguments
    num_shots_knitted = args.shots
    trotter_step = args.trotter_step
    debug_file = f"debug_step{trotter_step}_shots{num_shots_knitted}_{args.test_name}.txt"

    # Create Trotter step circuit
    circuit = trotter_stepper(trotter_step, Nqbits, epsilon, mass, insertion_point)
    config = ExperimentConfig(noise=True)

    run_and_log(circuit, num_shots_knitted, trotter_step, config, debug_file)


if __name__ == "__main__":
    main()
