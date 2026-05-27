#!/usr/bin/env python3
"""
Combined seed tests for second Trotter step with knitter.

Implements five tests from the tests matrix:
- 1 == 2: identical seeds should produce same fermion number and bootstrap error
- 1 != 3: different bootstrap seeds should produce different bootstrap errors but same fermion number
- 1 != 4: different simulator seeds should produce different fermion numbers
- 1 != 5: different transpiler seeds should produce different fermion numbers
- 4 != 5: different simulator and transpiler seeds should produce different fermion numbers

All runs use the second Trotter step circuit, noise=True, and 256 shots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from circuits.basic_circuits import trotter_stepper
from circuit_utils.statistics import fermion_number, bootstrap_error
from knitter.knitter import circuit_knitter
from config import ExperimentConfig


# Configuration parameters (from parent directory params.ipynb)
Nqbits = 12
epsilon = 0.8
mass = 1.125
insertion_point = 4  # Insertion point index for meson operator
num_shots_knitted = 256

# Create second Trotter step circuit (step 2)
circuit = trotter_stepper(2, Nqbits, epsilon, mass, insertion_point)

# Create a copy for the b runs
# circuit_copy = circuit.copy()


def run_knitter(simulator_seed=42, transpiler_seed=42, bootstrap_seed=42, noise=True):
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
    """Run the 1st run once and return its results."""
    print(f"\nPerforming 1st run (simulator_seed=42, transpiler_seed=42, bootstrap_seed=42)...")
    counts_run1 = run_knitter(simulator_seed=42, transpiler_seed=42, bootstrap_seed=42, noise=True)
    fn_run1 = fermion_number(counts_run1, insertion_point)
    boot_err_run1 = bootstrap_error(counts_run1, insertion_point, num_shots_knitted, seed=42)
    print(f"1st run fermion number: {fn_run1}")
    print(f"1st run bootstrap error: {boot_err_run1}")
    return counts_run1, fn_run1, boot_err_run1


# def run_1b_run():
#     """Run the 1b run with copied circuit, same parameters as run 1."""
#     print(f"\nPerforming 1b run (with copied circuit; simulator_seed=42, transpiler_seed=42, bootstrap_seed=42)...")
#     counts_run1b = run_knitter(circuit_input=circuit_copy, simulator_seed=42, transpiler_seed=42, bootstrap_seed=42, noise=True)
#     fn_run1b = fermion_number(counts_run1b, insertion_point)
#     boot_err_run1b = bootstrap_error(counts_run1b, insertion_point, num_shots_knitted, seed=42)
#     print(f"1b run fermion number: {fn_run1b}")
#     print(f"1b run bootstrap error: {boot_err_run1b}")
#     return counts_run1b, fn_run1b, boot_err_run1b
# 
# 
# def run_1b_eq_2b(fn_run1b, boot_err_run1b):
#     """1b == 2b: identical seeds on copied circuit should produce same fermion number and bootstrap error."""
#     # 2b run: same seeds 42, 42, 42 on copied circuit
#     print(f"\nPerforming 2b run (with copied circuit; simulator_seed=42, transpiler_seed=42, bootstrap_seed=42)...")
#     counts_run2b = run_knitter(circuit_input=circuit_copy, simulator_seed=42, transpiler_seed=42, bootstrap_seed=42, noise=True)
#     fn_run2b = fermion_number(counts_run2b, insertion_point)
#     boot_err_run2b = bootstrap_error(counts_run2b, insertion_point, num_shots_knitted, seed=42)
#     print(f"2b run fermion number: {fn_run2b}")
#     print(f"2b run bootstrap error: {boot_err_run2b}")
#     
#     fermion_match = fn_run1b == fn_run2b
#     bootstrap_match = boot_err_run1b == boot_err_run2b
#     
#     passed = fermion_match and bootstrap_match
#     return passed, "1b == 2b", fermion_match, bootstrap_match, fn_run1b, boot_err_run1b, fn_run2b, boot_err_run2b
# 
# 

def run_1_eq_2(fn_run1, boot_err_run1):
    """1 == 2: identical seeds should produce same fermion number and bootstrap error."""
    # 2nd run: same seeds 42, 42, 42
    print(f"\nPerforming 2nd run (simulator_seed=42, transpiler_seed=42, bootstrap_seed=42)...")
    counts_run2 = run_knitter(simulator_seed=42, transpiler_seed=42, bootstrap_seed=42, noise=True)
    fn_run2 = fermion_number(counts_run2, insertion_point)
    boot_err_run2 = bootstrap_error(counts_run2, insertion_point, num_shots_knitted, seed=42)
    print(f"2nd run fermion number: {fn_run2}")
    print(f"2nd run bootstrap error: {boot_err_run2}")
    
    fermion_match = fn_run1 == fn_run2
    bootstrap_match = boot_err_run1 == boot_err_run2
    
    passed = fermion_match and bootstrap_match
    return passed, "1 == 2", fermion_match, bootstrap_match, fn_run1, boot_err_run1, fn_run2, boot_err_run2


def run_1_neq_3(fn_run1, boot_err_run1):
    """1 != 3: different bootstrap seeds should produce different bootstrap errors but same fermion number."""
    # 3rd run: seeds 42, 42, 123
    print(f"\nPerforming 3rd run (simulator_seed=42, transpiler_seed=42, bootstrap_seed=123)...")
    counts_run3 = run_knitter(simulator_seed=42, transpiler_seed=42, bootstrap_seed=123, noise=True)
    fn_run3 = fermion_number(counts_run3, insertion_point)
    boot_err_run3 = bootstrap_error(counts_run3, insertion_point, num_shots_knitted, seed=123)
    print(f"3rd run fermion number: {fn_run3}")
    print(f"3rd run bootstrap error: {boot_err_run3}")
    
    fermion_match = fn_run1 == fn_run3
    bootstrap_differs = boot_err_run1 != boot_err_run3
    passed = fermion_match and bootstrap_differs
    return passed, "1 != 3", fermion_match, bootstrap_differs, fn_run1, boot_err_run1, fn_run3, boot_err_run3


def run_1_neq_4(fn_run1, boot_err_run1):
    """1 != 4: different simulator seeds should produce different fermion numbers."""
    # 4th run: seeds 123, 42, 42
    print(f"\nPerforming 4th run (simulator_seed=123, transpiler_seed=42, bootstrap_seed=42)...")
    counts_run4 = run_knitter(simulator_seed=123, transpiler_seed=42, bootstrap_seed=42, noise=True)
    fn_run4 = fermion_number(counts_run4, insertion_point)
    boot_err_run4 = bootstrap_error(counts_run4, insertion_point, num_shots_knitted, seed=42)
    print(f"4th run fermion number: {fn_run4}")
    print(f"4th run bootstrap error: {boot_err_run4}")
    
    passed = fn_run1 != fn_run4
    return passed, "1 != 4", fn_run1, boot_err_run1, fn_run4, boot_err_run4


def run_1_neq_5(fn_run1, boot_err_run1):
    """1 != 5: different transpiler seeds should produce different fermion numbers."""
    # 5th run: seeds 42, 123, 42
    print(f"\nPerforming 5th run (noisy; simulator_seed=42, transpiler_seed=123, bootstrap_seed=42; shots={num_shots_knitted})...")
    counts_run5 = run_knitter(simulator_seed=42, transpiler_seed=123, bootstrap_seed=42, noise=True)
    fn_run5 = fermion_number(counts_run5, insertion_point)
    boot_err_run5 = bootstrap_error(counts_run5, insertion_point, num_shots_knitted, seed=42)
    print(f"5th run fermion number: {fn_run5}")
    print(f"5th run bootstrap error: {boot_err_run5}")
    
    passed = fn_run1 != fn_run5
    return passed, "1 != 5", fn_run1, boot_err_run1, fn_run5, boot_err_run5


def run_4_neq_5(fn_run4, fn_run5):
    """4 != 5: different simulator and transpiler seeds should produce different fermion numbers."""
    print(f"\nComparing runs 4 and 5 fermion numbers...")
    passed = fn_run4 != fn_run5
    return passed, "4 != 5", fn_run4, fn_run5


if __name__ == "__main__":
    print(f"Running all seed tests for second Trotter step (noisy, shots={num_shots_knitted})")
    print("=" * 70)
    
    results = []
    
    # Run the 1st run once and reuse results
    counts_run1, fn_run1, boot_err_run1 = run_1st_run()
    
    # # Run the 1b run with copied circuit
    # counts_run1b, fn_run1b, boot_err_run1b = run_1b_run()
    # 
    # # Test 1b == 2b with copied circuit
    # passed, name, fermion_match, bootstrap_match, fn1b, be1b, fn2b, be2b = run_1b_eq_2b(fn_run1b, boot_err_run1b)
    # if passed:
    #     results.append((name, True, f"Fermion numbers and bootstrap errors match: {fn1b} == {fn2b}, {be1b} == {be2b}"))
    # else:
    #     details = []
    #     if not fermion_match:
    #         details.append(f"fermion numbers differ: {fn1b} vs {fn2b}")
    #     if not bootstrap_match:
    #         details.append(f"bootstrap errors differ: {be1b} vs {be2b}")
    #     results.append((name, False, "; ".join(details)))
    
    passed, name, fermion_match, bootstrap_match, fn1, be1, fn2, be2 = run_1_eq_2(fn_run1, boot_err_run1)
    if passed:
        results.append((name, True, f"Fermion numbers and bootstrap errors match: {fn1} == {fn2}, {be1} == {be2}"))
    else:
        details = []
        if not fermion_match:
            details.append(f"fermion numbers differ: {fn1} vs {fn2}")
        if not bootstrap_match:
            details.append(f"bootstrap errors differ: {be1} vs {be2}")
        results.append((name, False, "; ".join(details)))
    
    passed, name, fermion_match, bootstrap_differs, fn1, be1, fn3, be3 = run_1_neq_3(fn_run1, boot_err_run1)
    if passed:
        results.append((name, True, f"Fermion numbers match and bootstrap errors differ: {fn1} == {fn3}, {be1} != {be3}"))
    else:
        details = []
        if not fermion_match:
            details.append(f"fermion numbers differ: {fn1} vs {fn3}")
        if not bootstrap_differs:
            details.append(f"bootstrap errors are same: {be1}")
        results.append((name, False, "; ".join(details)))
    
    passed, name, fn1, be1, fn4, be4 = run_1_neq_4(fn_run1, boot_err_run1)
    if passed:
        results.append((name, True, f"Fermion numbers differ: {fn1} vs {fn4}"))
    else:
        results.append((name, False, f"Fermion numbers are same: {fn1}"))
    
    passed, name, fn1, be1, fn5, be5 = run_1_neq_5(fn_run1, boot_err_run1)
    if passed:
        results.append((name, True, f"Fermion numbers differ: {fn1} vs {fn5}"))
    else:
        results.append((name, False, f"Fermion numbers are same: {fn1}"))
    
    passed, name, fn4, fn5 = run_4_neq_5(fn4, fn5)
    if passed:
        results.append((name, True, f"Fermion numbers differ: {fn4} vs {fn5}"))
    else:
        results.append((name, False, f"Fermion numbers are same: {fn4}"))
    
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
