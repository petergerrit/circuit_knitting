#!/usr/bin/env python
"""
Test suite for simulator randomness.

This script tests that the simulator produces different results with different seeds
and identical results with the same seed, using a simple circuit that demonstrates
quantum randomness clearly.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from config import ExperimentConfig
from experiment import run_circuit_experiment


class TestSimulatorRandomness(unittest.TestCase):
    """Test cases for simulator randomness."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test configuration with no noise
        self.config = ExperimentConfig(
            noise=False,
            num_shots=1024,  # Sufficient shots for statistical significance
            results_dir="test_results"
        )
        
        # Create a simple circuit that will show clear quantum randomness
        # This circuit puts all qubits in superposition and measures them
        self.test_circuit = QuantumCircuit(3, 3)
        self.test_circuit.h(0)  # Hadamard gate creates superposition
        self.test_circuit.h(1)
        self.test_circuit.h(2)
        self.test_circuit.measure([0, 1, 2], [0, 1, 2])
        
        # Display circuit diagram
        print("\nTest Circuit Diagram:")
        print(self.test_circuit.draw(output='text'))
        
    def test_simulator_randomness(self):
        """Test that simulator produces different results with different seeds and same with same seed."""
        print("\nTesting simulator randomness...")
        print("Using 1024 shots with a simple superposition circuit")
        
        # First evaluation with seed 42
        seed1 = 42
        print(f"\nEvaluation 1 with seed {seed1}")
        results1 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config,
            simulator_seed=seed1,
            transpiler_seed=seed1
        )
        print(f"Results 1: {results1}")
        
        # Second evaluation with different seed (123)
        seed2 = 123
        print(f"\nEvaluation 2 with seed {seed2}")
        results2 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config,
            simulator_seed=seed2,
            transpiler_seed=seed2
        )
        print(f"Results 2: {results2}")
        
        # Third evaluation with original seed (42) - should match results1
        print(f"\nEvaluation 3 with seed {seed1} (same as evaluation 1)")
        results3 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config,
            simulator_seed=seed1,
            transpiler_seed=seed1
        )
        print(f"Results 3: {results3}")
        
        # Verify reproducibility: results1 should equal results3
        self.assertEqual(results1, results3, 
                        "Same seed should produce identical results")
        print("✓ Reproducibility confirmed: results1 == results3")
        
        # Verify different seeds produce different results
        self.assertNotEqual(results1, results2, 
                           "Different seeds should produce different results")
        print("✓ Randomness confirmed: results1 != results2")
        
        # Additional check: results3 should also differ from results2
        self.assertNotEqual(results3, results2, 
                           "Different seeds should produce different results (3 vs 2)")
        print("✓ Randomness confirmed: results3 != results2")
        
        print("\n✓ All simulator randomness tests passed!")


def run_randomness_tests():
    """Run randomness tests and generate a summary."""
    print("Running Simulator Randomness Tests...")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSimulatorRandomness)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 55)
    print(f"Randomness tests completed: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ Simulator randomness verified!")
        print("  - Same seeds produce identical results")
        print("  - Different seeds produce different results")
    else:
        print("✗ Simulator randomness tests failed!")
    
    return result


if __name__ == "__main__":
    # Run randomness tests
    test_result = run_randomness_tests()
    
    # Exit with appropriate code
    exit(0 if test_result.wasSuccessful() else 1)