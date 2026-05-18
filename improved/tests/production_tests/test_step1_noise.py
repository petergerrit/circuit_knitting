#!/usr/bin/env python3
"""
Test 1 != 2 from tests matrix: noise effect on first Trotter step with knitter.

Compares fermion number from noiseless vs noisy knitter runs.
Both use seeds (42, 42, 42) and 128 shots.
Expects: fermion numbers should be different (1 != 2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number
from knitter.knitter import circuit_knitter
from config import ExperimentConfig


# Configuration parameters (from parent directory params.ipynb)
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4  # Insertion point index for meson operator
num_shots_knitted = 128

# Create first Trotter step circuit (step 1)
circuit = trotter_stepper(1, Nqbits, epsilon, mass, insertion_point)


def run_knitter(noise, simulator_seed=42, transpiler_seed=42):
    """Run knitter with given parameters, return counts."""
    config = ExperimentConfig(noise=noise)
    
    knitted_results = circuit_knitter(
        circuit=circuit,
        start_qubit=0,
        end_qubit=10,
        num_shots=num_shots_knitted,
        config=config,
        simulator_seed=simulator_seed,
        transpiler_seed=transpiler_seed
    )
    
    return knitted_results['results']


def run_1st_run():
    """Run the 1st run (noiseless) once and return its results."""
    print(f"\nPerforming 1st run (noiseless; simulator_seed=42, transpiler_seed=42, shots={num_shots_knitted})...")
    counts_run1 = run_knitter(noise=False, simulator_seed=42, transpiler_seed=42)
    fn_run1 = fermion_number(counts_run1, insertion_point)
    print(f"1st run fermion number: {fn_run1}")
    return counts_run1, fn_run1


def run_2nd_run():
    """Run the 2nd run (noisy) and return its results."""
    print(f"\nPerforming 2nd run (noisy; simulator_seed=42, transpiler_seed=42, shots={num_shots_knitted})...")
    counts_run2 = run_knitter(noise=True, simulator_seed=42, transpiler_seed=42)
    fn_run2 = fermion_number(counts_run2, insertion_point)
    print(f"2nd run fermion number: {fn_run2}")
    return counts_run2, fn_run2


if __name__ == "__main__":
    print(f"Running noise effect test for first Trotter step (shots={num_shots_knitted})")
    print("=" * 70)
    
    results = []
    
    # Run both tests
    counts_run1, fn_run1 = run_1st_run()
    counts_run2, fn_run2 = run_2nd_run()
    
    passed = fn_run1 != fn_run2
    name = "1 != 2"
    if passed:
        results.append((name, True, f"Fermion numbers differ: {fn_run1} vs {fn_run2}"))
    else:
        results.append((name, False, f"Fermion numbers are same: {fn_run1}"))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed, details in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name} - {details}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    sys.exit(0 if all_passed else 1)
