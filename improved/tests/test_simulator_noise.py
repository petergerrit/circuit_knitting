#!/usr/bin/env python
"""
Minimal test for circuit randomness using the simplest possible circuit.

This test uses a very simple 2-qubit circuit with just Hadamard gates and measurements
to test the basic randomness functionality with and without noise.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from config import ExperimentConfig
from experiment import run_circuit_experiment


class TestMinimalNoise(unittest.TestCase):
    """Test cases for minimal circuit randomness with noise comparison."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configurations for no noise and with noise
        self.config_no_noise = ExperimentConfig(
            noise=False,
            num_shots=32,  # Very low shots for minimal testing
            results_dir="test_results"
        )
        
        self.config_with_noise = ExperimentConfig(
            noise=True,
            num_shots=32,  # Very low shots for minimal testing
            results_dir="test_results",
            optimization_level=1  # Match knitter's optimization level
        )
        
        # Create the simplest possible circuit: Hadamard gates, CNOT, and measurements
        self.test_circuit = QuantumCircuit(2, 2)
        self.test_circuit.h(0)  # Hadamard gate creates superposition
        self.test_circuit.h(1)
        self.test_circuit.cx(0, 1)  # CNOT gate
        self.test_circuit.measure([0, 1], [0, 1])
    
    def test_minimal_noise_comparison(self):
        """Test noisy vs noiseless execution with the same circuit."""
        print("\nMinimal Test Circuit Diagram:")
        print(self.test_circuit.draw(output='text'))
        print("\nTesting minimal circuit with noise comparison...")
        print("Using 32 shots with simplest possible circuit")
        
        # Evaluation 1: Noisy with seed 42
        print(f"\nEvaluation 1 (noisy, seed 42)")
        results1 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config_with_noise,
            simulator_seed=42,
            transpiler_seed=42
        )
        print(f"Results 1: {results1}")
        
        # Evaluation 2: Noisy with seed 123
        print(f"\nEvaluation 2 (noisy, seed 123)")
        results2 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config_with_noise,
            simulator_seed=123,
            transpiler_seed=123
        )
        print(f"Results 2: {results2}")
        
        # Evaluation 3: Noisy with seed 42 (should match results1)
        print(f"\nEvaluation 3 (noisy, seed 42)")
        results3 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config_with_noise,
            simulator_seed=42,
            transpiler_seed=42
        )
        print(f"Results 3: {results3}")
        
        # Evaluation 4: Noiseless with seed 42
        print(f"\nEvaluation 4 (noiseless, seed 42)")
        results4 = run_circuit_experiment(
            circuit=self.test_circuit,
            config=self.config_no_noise,
            simulator_seed=42,
            transpiler_seed=42
        )
        print(f"Results 4: {results4}")
        
        # Verify reproducibility: results1 should equal results3
        self.assertEqual(results1, results3, 
                        "Same seed with noise should produce identical results")
        print("✓ Reproducibility confirmed: results1 == results3")
        
        # Verify different seeds produce different results
        self.assertNotEqual(results1, results2, 
                           "Different seeds with noise should produce different results")
        print("✓ Randomness confirmed: results1 != results2")
        
        # Verify noise makes a difference
        self.assertNotEqual(results1, results4, 
                           "Noisy and noiseless results should be different")
        print("✓ Noise effect confirmed: results1 != results4")
        
        print("\n✓ All minimal noise comparison tests passed!")


def run_minimal_noise_tests():
    """Run minimal noise tests and generate a summary."""
    print("Running Minimal Noise Comparison Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMinimalNoise)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Minimal noise tests completed: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ Minimal noise comparison verified!")
        print("  - Same seeds with noise produce identical results")
        print("  - Different seeds with noise produce different results")
        print("  - Noisy and noiseless results are different")
        print("  - Using simplest possible circuit (2 qubits, Hadamard + measure)")
    else:
        print("✗ Minimal noise comparison tests failed!")
    
    return result


if __name__ == "__main__":
    # Run minimal noise tests
    test_result = run_minimal_noise_tests()
    
    # Exit with appropriate code
    exit(0 if test_result.wasSuccessful() else 1)
