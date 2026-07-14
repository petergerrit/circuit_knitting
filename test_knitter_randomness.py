#!/usr/bin/env python3
"""
Test script to verify that the knitter produces different results when no seed is provided,
and consistent results when a seed is provided.
"""

import sys
import os
sys.path.append('/home/peter/git/circuit_knitting/improved')

from qiskit import QuantumCircuit
from experiment import circuit_knitter, ExperimentConfig

def test_knitter_randomness():
    """Test that knitter produces different results without seed, same results with seed."""
    
    # Create a simple test circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    config = ExperimentConfig(noise=False, optimization_level=1, num_shots=100, results_dir='test_results')
    
    print("Testing knitter randomness...")
    
    # Test 1: Run without seed multiple times - should get different results
    print("\n1. Testing without seed (should produce different results):")
    results_without_seed = []
    for i in range(3):
        result = circuit_knitter(qc, start_qubit=0, end_qubit=1, num_shots=100, config=config)
        results_without_seed.append(result['results'])
        print(f"  Run {i+1}: {result['results']}")
    
    # Check if results are different
    all_same = all(r == results_without_seed[0] for r in results_without_seed)
    print(f"  All results identical: {all_same}")
    if all_same:
        print("  ❌ FAIL: Results should be different without seed")
        return False
    else:
        print("  ✅ PASS: Results are different without seed")
    
    # Test 2: Run with same seed multiple times - should get identical results
    print("\n2. Testing with same seed (should produce identical results):")
    results_with_seed = []
    seed = 42
    for i in range(3):
        result = circuit_knitter(qc, start_qubit=0, end_qubit=1, num_shots=100, config=config, 
                                 simulator_seed=seed, transpiler_seed=seed)
        results_with_seed.append(result['results'])
        print(f"  Run {i+1} with seed {seed}: {result['results']}")
    
    # Check if results are identical
    all_same = all(r == results_with_seed[0] for r in results_with_seed)
    print(f"  All results identical: {all_same}")
    if all_same:
        print("  ✅ PASS: Results are identical with same seed")
    else:
        print("  ❌ FAIL: Results should be identical with same seed")
        return False
    
    # Test 3: Run with different seeds - should get different results
    print("\n3. Testing with different seeds (should produce different results):")
    result_seed1 = circuit_knitter(qc, start_qubit=0, end_qubit=1, num_shots=100, config=config, 
                                   simulator_seed=42, transpiler_seed=42)
    result_seed2 = circuit_knitter(qc, start_qubit=0, end_qubit=1, num_shots=100, config=config, 
                                   simulator_seed=123, transpiler_seed=123)
    
    print(f"  Result with seed 42: {result_seed1['results']}")
    print(f"  Result with seed 123: {result_seed2['results']}")
    
    different = result_seed1['results'] != result_seed2['results']
    print(f"  Results different: {different}")
    if different:
        print("  ✅ PASS: Results are different with different seeds")
    else:
        print("  ❌ FAIL: Results should be different with different seeds")
        return False
    
    print("\n🎉 All tests passed! The knitter now works correctly.")
    return True

if __name__ == "__main__":
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    test_knitter_randomness()